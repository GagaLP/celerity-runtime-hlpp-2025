#pragma once

#include <vector>

#include "backend/backend.h"
#include "runtime.h"
#include "system_info.h"
#include "task.h"


// EXTREMELY HACKY T0 PROOF-OF-CONCEPT IMPLEMENTATION

namespace celerity::detail::scratch_buffer_detail {

struct scratch_allocation {
	range<3> buffer_range;
	device_id did;
	void* ptr;
};

class HACK_scratch_buffer_registry {
  public:
	static HACK_scratch_buffer_registry& get_instance() {
		static HACK_scratch_buffer_registry instance;
		return instance;
	}

	buffer_id register_buffer(std::vector<scratch_allocation> allocs) {
		// FIXME: Ids overlap with normal buffers
		const auto bid = m_next_bid++;
		[[maybe_unused]] const auto [_, inserted] = m_allocs.emplace(bid, std::move(allocs));

		assert(inserted);
		return bid;
	}

	std::vector<scratch_allocation> destroy_buffer(const buffer_id bid) {
		const auto allocs = m_allocs.at(bid);
		m_allocs.erase(bid);
		return allocs;
	}

	const std::vector<scratch_allocation>& get_allocations(const buffer_id bid) {
		if(m_allocs.find(bid) == m_allocs.end()) {
			throw std::runtime_error("No allocation found for scratch buffer, check the scope of this scratch buffer.");
		}
		return m_allocs.at(bid);
	}

  private:
	buffer_id m_next_bid = 0;
	std::unordered_map<buffer_id, std::vector<scratch_allocation>> m_allocs = {};
};
} // namespace celerity::detail::scratch_buffer_detail

namespace celerity {

enum class scratch_buffer_scope {
	chunk,
	device,
	// node // TODO API: Do we ever want this..? Does SYCL (or CUDA) support cross-device atomics..?
	// => Turns out yes, we want this: E.g. for global tile counts in UMUGUC. (Technically it's unlikely that
	// all chunks require the full buffer, but there's probably no point in splitting that up further).
	// => We could allocate node-scoped scratch buffers using managed memory (=> benchmark perf though!)
	//    and maybe forbid concurrent write access..?
};

template <typename DataT, int Dims = 1>
class scratch_buffer {
  public:
	// TODO API: Size should not have to be uniform across all chunks/devices
	//           Maybe do a range and "window" (to support global indexing)
	scratch_buffer(const range<Dims>& range, const scratch_buffer_scope scope = scratch_buffer_scope::chunk) {
		using namespace detail;
		if(scope == scratch_buffer_scope::device) { throw std::runtime_error("Not quite sure yet what that even means"); }

		// const node_id this_nid = detail::runtime::get_instance().NOCOMMIT_get_local_nid();
		auto* backend_ptr = detail::runtime::get_instance().NOCOMMIT_get_backend_ptr();
		assert(backend_ptr != nullptr);

		const auto num_devices = backend_ptr->get_system_info().devices.size();

		std::vector<scratch_buffer_detail::scratch_allocation> allocs;
		for(int did = 0; did < num_devices; ++did) {
			// Kind of illegal to call this from the main thread, but since we're just forwarding the call to SYCL it *should* be fine
			auto event = backend_ptr->enqueue_device_alloc(did, range.size() * sizeof(DataT), alignof(DataT));
			while(!event.is_complete()) {}
			auto ptr = event.get_result();
			allocs.emplace_back(detail::range_cast<3>(range), did, ptr);
			// CELERITY_CRITICAL("Pointer for scratch buffer of size {} on device {}: {}. Alloc size is {}*{} = {} bytes", range, did, ptr, range.size(),
			//     sizeof(DataT), range.size() * sizeof(DataT));
		}

		m_bid = scratch_buffer_detail::HACK_scratch_buffer_registry::get_instance().register_buffer(std::move(allocs));
		// CELERITY_CRITICAL("Pointer bid {}", m_bid);
	}

	// HACK required b/c in debug builds allocations are initialized w/ debug pattern.
	// (Also we need to reset the point counter between UMUGUC count and write kernels. could use a separate buffer in that case though)
	// TODO API: If we instead wanted to manually launch a kernel (on user side) to fill the buffer, we'd have the problem that it
	// would (in case for UMUGUC) require a different geometry!
	void fill(const int value) {
		const auto& allocs = detail::scratch_buffer_detail::HACK_scratch_buffer_registry::get_instance().get_allocations(m_bid);
		auto* backend_ptr = detail::runtime::get_instance().NOCOMMIT_get_backend_ptr();
		assert(backend_ptr != nullptr);
		for(const auto& [buffer_range, did, ptr] : allocs) {
			auto event = backend_ptr->enqueue_device_memset(did, ptr, value, buffer_range.size() * sizeof(DataT));
			while(!event.is_complete()) {}
		}
	}

	// TODO API: Should this be possible in main thread, or just host task?
	// TODO API: This should probably also include some information about the chunk
	// TODO API: Return a buffer_snapshot instead
	std::vector<std::vector<DataT>> get_data_on_host() {
		using namespace detail;
		std::vector<std::vector<DataT>> result;
		auto* backend_ptr = detail::runtime::get_instance().NOCOMMIT_get_backend_ptr();
		assert(backend_ptr != nullptr);
		const auto& allocs = scratch_buffer_detail::HACK_scratch_buffer_registry::get_instance().get_allocations(m_bid);
		for(const auto& [buffer_range, did, ptr] : allocs) {
			std::vector<DataT> dest(buffer_range.size());
			const box<3> copy_box({}, buffer_range);
			auto event =
			    backend_ptr->enqueue_device_copy(did, 0, ptr, dest.data(), detail::linearized_layout{0}, detail::linearized_layout{0}, copy_box, sizeof(DataT));
			while(!event.is_complete()) {}
			result.emplace_back(std::move(dest));
		}
		return result;
	}

	void set_data_from_host(const std::vector<std::vector<DataT>>& data) {
		using namespace detail;
		auto* backend_ptr = detail::runtime::get_instance().NOCOMMIT_get_backend_ptr();
		assert(backend_ptr != nullptr);
		const auto& allocs = scratch_buffer_detail::HACK_scratch_buffer_registry::get_instance().get_allocations(m_bid);

		if(data.size() != allocs.size()) { throw std::runtime_error("Data / allocation count mismatch"); }
		for(size_t i = 0; i < data.size(); ++i) {
			const auto& [buffer_range, did, ptr] = allocs[i];
			if(data[i].size() != buffer_range.size()) { throw std::runtime_error("Data / allocation size mismatch"); }
			const box<3> copy_box({}, buffer_range);
			auto event = backend_ptr->enqueue_device_copy(
			    did, 0, data[i].data(), ptr, detail::linearized_layout{0}, detail::linearized_layout{0}, copy_box, sizeof(DataT));
			while(!event.is_complete()) {}
		}
	}

	~scratch_buffer() {
		using namespace detail;
		auto* backend_ptr = detail::runtime::get_instance().NOCOMMIT_get_backend_ptr();
		assert(backend_ptr != nullptr);
		auto chunk_ptrs = scratch_buffer_detail::HACK_scratch_buffer_registry::get_instance().destroy_buffer(m_bid);
		for(const auto& [_2, did, ptr] : chunk_ptrs) {
			auto event = backend_ptr->enqueue_device_free(did, ptr);
			while(!event.is_complete()) {}
		}
	}

	detail::buffer_id get_id() const { return m_bid; }

  private:
	detail::buffer_id m_bid;
};

// TENTATIVE PLAN:
// - For hydration, look up chunk dimensions (PROBLEM: What about overlapping chunks? How can we uniquely identify those?!)
// - Store all pointers in a global singleton, index by scratch buffer id (TODO), read it from there

template <typename DataT, int Dims = 1>
class scratch_accessor {
  public:
	scratch_accessor(const scratch_buffer<DataT, Dims>& buffer) : m_bid(buffer.get_id()) {}

	scratch_accessor(const scratch_accessor& other)
	    : m_bid(other.m_bid), m_ptr(other.m_ptr), m_allocation_range(other.m_allocation_range), m_allocation_offset(other.m_allocation_offset) {
#if !defined(__SYCL_DEVICE_ONLY__)
		if(detail::closure_hydrator::is_available() && detail::closure_hydrator::get_instance().is_hydrating()) {
			const auto sr = detail::closure_hydrator::get_instance().get_execution_range();
			const auto did = detail::closure_hydrator::get_instance().get_device_id();
			const auto& alloc = detail::scratch_buffer_detail::HACK_scratch_buffer_registry::get_instance().get_allocations(m_bid)[did];
			m_ptr = static_cast<DataT*>(alloc.ptr);
			m_allocation_range = detail::range_cast<Dims>(alloc.buffer_range);
			// CELERITY_CRITICAL("HYDRATED scratch_accessor for buffer {} with size {} for chunk {} with ptr {}", m_bid, m_allocation_range, sr, (void*)m_ptr);
		}
#endif
	}

	DataT& operator[](const id<Dims>& index) const { return m_ptr[get_linear_offset(index)]; }

	//   private:
	detail::buffer_id m_bid;
	DataT* m_ptr;
	range<Dims> m_allocation_range;
	id<Dims> m_allocation_offset = detail::zeros; // NYI

	size_t get_linear_offset(const id<Dims>& index) const { return detail::get_linear_index(m_allocation_range, index); }
};

} // namespace celerity
