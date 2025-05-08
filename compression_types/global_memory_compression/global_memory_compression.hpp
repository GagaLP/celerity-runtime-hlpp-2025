#pragma once

#include <celerity.h>
#include <scratch_buffer.h>

#include <algorithm>

namespace celerity {
template <typename T, typename Q>
class compressed<celerity::compression::point_cloud<T, Q>> {
	using compression_type = typename celerity::compression::point_cloud<T, Q>::compression_type;
	using value_type = typename celerity::compression::point_cloud<T, Q>::value_type;

  public:
	compressed() = default;

	inline const size_t get_index(auto range, auto global_id, auto tile_size, celerity::range<3> offset = {0, 0, 0}) const {
		return range.get(0) * tile_size * (global_id[0] + offset[0]) + tile_size * (global_id[1] + offset[1]);
	}

	template <typename GlobalOffset, typename CompressedPoints, typename DataAvailable, typename UncompressedData>
	void compress(celerity::nd_item<3> item, DataAvailable& data_point_available, const UncompressedData& local_tile_point_acc,
	    GlobalOffset& compression_offset_acc, CompressedPoints& tile_point_acc, int work_items_per_tile, int points_per_tile, T min, int tile_size) const {
		if(item.get_global_id(2) >= points_per_tile) { return; }

		int actual_points = data_point_available[{item.get_global_id(0), item.get_global_id(1)}];

		size_t amount = actual_points / work_items_per_tile;
		size_t num_points_per_runner = (amount + 1);

		T min_max = {min.x() + tile_size * item.get_global_id(0), min.y() + tile_size * item.get_global_id(1), 0};

		auto center = min_max;

		compression_offset_acc[{item.get_global_id(0), item.get_global_id(1), 0}] = center;

		for(size_t i = 0; i < num_points_per_runner; i++) {
			size_t idx = i + (item.get_local_id(2) * num_points_per_runner);

			if(idx >= points_per_tile) { break; }

			T p = local_tile_point_acc[{get_index(item.get_global_range(), item.get_global_id(), points_per_tile, {0, 0, 0}) + idx}];

			p = p - center;

			compression_type compressed_point = {p.x(), p.y(), p.z()};

			tile_point_acc[{item.get_global_id(0), item.get_global_id(1), idx}] = compressed_point;
		}
	}

	template <typename GlobalPoints, typename LocalPoints, typename Point>
	void decompress_memory_chunk(int x, int y, celerity::nd_item<3>& item, int num_points_per_runner, int actual_points, int points_per_tile,
	    Point& centerpoint, GlobalPoints& tile_point_acc_last_dim, LocalPoints& local_tile_point_acc) const {
		for(int i = 0; i < num_points_per_runner; i++) {
			size_t idx = i + (item.get_global_id(2) * num_points_per_runner);

			if(actual_points <= idx) { break; }

			const Q p = tile_point_acc_last_dim[{(item.get_global_id(0) + x), (item.get_global_id(1) + y), idx}];

			T new_p;
			new_p.x() = p.x() + centerpoint.x();
			new_p.y() = p.y() + centerpoint.y();
			new_p.z() = p.z() + centerpoint.z();

			local_tile_point_acc[{
			    get_index(item.get_global_range(), item.get_global_id(), points_per_tile, {static_cast<size_t>(x), static_cast<size_t>(y), 0}) + idx}] = new_p;
		}
	}


	template <typename GlobalOffset, typename CompressedData, typename UncompressedData, typename DataAvailable>
	void decompress(GlobalOffset& compression_offset_acc, CompressedData& compressed_data_acc, UncompressedData& uncompressed_data_acc,
	    celerity::nd_item<3> item, const int points_per_tile, const int work_items_per_tile, DataAvailable& data_point_available, const int x,
	    const int y) const {
		int amount = points_per_tile / work_items_per_tile;
		int num_points_per_runner = (amount + 1);

		if(item.get_global_id(0) + x >= 0 && item.get_global_id(0) + x < item.get_global_range(0) && item.get_global_id(1) + y >= 0
		    && item.get_global_id(1) + y < item.get_global_range(1)) {
			int actual_points = data_point_available[{item.get_global_id(0) + x, item.get_global_id(1) + y}];

			const auto& cp = compression_offset_acc[{item.get_global_id(0) + x, item.get_global_id(1) + y, 0}];
			decompress_memory_chunk(x, y, item, num_points_per_runner, actual_points, points_per_tile, cp, compressed_data_acc, uncompressed_data_acc);
		}
	}

	template <typename CompressedData, typename GlobalMetadata, typename GlobalCount, typename UncompressedData>
	void decompress(CompressedData& compressed_data, GlobalMetadata& m_global_offset, GlobalCount& data_point_available, UncompressedData& uncompressed_data,
	    const size_t width, const size_t height) const {
		for(size_t i = 0; i < width; i++) {
			for(size_t j = 0; j < height; j++) {
				int actual_points = data_point_available[{i, j}];
				for(size_t k = 0; k < actual_points; k++) {
					compression_type p = compressed_data[{i, j, k}];
					value_type compression_offset = m_global_offset[{i, j, 0}];

					value_type decompressed_p;
					decompressed_p.x() = p.x() + compression_offset.x();
					decompressed_p.y() = p.y() + compression_offset.y();
					decompressed_p.z() = p.z() + compression_offset.z();

					uncompressed_data[i][j][k] = decompressed_p;
				}
			}
		}
	}
};


template <int TargetDims, typename Target, int SubscriptDim = 0>
class subscript_proxy_compressed;

template <int TargetDims, typename Target, int SubscriptDim>
inline decltype(auto) subscript_compressed(Target& tgt, id<TargetDims> id, const size_t index, nd_item<TargetDims> item, const int const_offset) {
	static_assert(SubscriptDim < TargetDims);
	id[SubscriptDim] = index - item.get_global_id().get(SubscriptDim) + const_offset;
	if constexpr(SubscriptDim == TargetDims - 1) {
		return tgt[std::as_const(id[2])];
	} else {
		return subscript_proxy_compressed<TargetDims, Target, SubscriptDim + 1>{tgt, id, item, const_offset};
	}
}

template <int TargetDims, typename Target>
inline decltype(auto) subscript_compressed(Target& tgt, const size_t index, nd_item<TargetDims> item, const int const_offset) {
	return subscript_compressed<TargetDims, Target, 0>(tgt, id<TargetDims>{}, index);
}

template <int TargetDims, typename Target, int SubscriptDim>
class subscript_proxy_compressed {
  public:
	subscript_proxy_compressed(Target& tgt, const id<TargetDims> id, nd_item<TargetDims> item, const int const_offset)
	    : m_tgt(tgt), m_id(id), m_item(item), m_const_offset(const_offset) {}

	inline decltype(auto) operator[](const size_t index) const { //
		return subscript_compressed<TargetDims, Target, SubscriptDim>(m_tgt, m_id, index, m_item, m_const_offset);
	}

  private:
	Target& m_tgt;
	id<TargetDims> m_id{};
	nd_item<TargetDims> m_item;
	const int m_const_offset;
};

template <access_mode AccessMode, typename DataT, int Dim, typename Compression, typename GlobalOffset, typename CompressedData, typename UncompressedData,
    typename DataAvailable, typename DecompressionInterface, typename LocalDecompressionInterface>
struct local_accessor_compressor {
	local_accessor_compressor(const Compression& compression, const GlobalOffset& compression_offset_acc, CompressedData& compressed_data_acc,
	    UncompressedData uncompressed_data_acc, nd_item<Dim> item, const int points_per_tile, const int work_items_per_tile,
	    DataAvailable& data_point_available, const int x, const int y, const DataT min, const int tile_size,
	    DecompressionInterface& local_decompression_interface, LocalDecompressionInterface& local_decompression_interface_local)
	    : m_item(item), m_compression(compression), m_data_point_available(data_point_available), m_local_tile(uncompressed_data_acc),
	      m_work_items_per_tile(work_items_per_tile), m_points_per_tile(points_per_tile), m_compression_offset_acc(compression_offset_acc),
	      m_compressed_data(compressed_data_acc), m_x(x), m_y(y), m_min(min), m_tile_size(tile_size),
	      m_local_decompression_interface(local_decompression_interface), m_local_decompression_interface_local(local_decompression_interface_local) {
		if constexpr(detail::is_consumer_mode(AccessMode)) {
			if(item.get_global_id(0) + m_x < item.get_global_range(0) && item.get_global_id(1) + m_y < item.get_global_range(1)) {
				sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
				    m_local_decompression_interface[{item.get_global_id(0) + m_x, item.get_global_id(1) + m_y, 0}]};

				int32_t status = 0;
				int32_t count = 0;

				if(item.get_global_id(2) == 0) {
					int32_t combined_status = atomic_ref_count.load();

					do {
						auto [separate_status, separate_count] = separate_status_atomic(combined_status);

						status = separate_status;
						count = separate_count;

						count++;

						if(status == compressing) {
							do {
								int32_t combined_status = atomic_ref_count.load();

								auto [separate_status, separate_count] = separate_status_atomic(combined_status);

								status = separate_status;
								count = separate_count;
							} while(status == compressing);
						}

						if(status == is_compressed && count == 1) {
							status = decompressing;
							m_local_decompression_interface_local[0] = decompressing;
						}
					} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
				}
			}

			celerity::group_barrier(item.get_group());

			if(m_local_decompression_interface_local[0] == decompressing) {
				m_compression.decompress(
				    compression_offset_acc, compressed_data_acc, uncompressed_data_acc, item, points_per_tile, work_items_per_tile, data_point_available, x, y);
			}

			celerity::group_barrier(item.get_group());

			if(item.get_global_id(0) + m_x < item.get_global_range(0) && item.get_global_id(1) + m_y < item.get_global_range(1)) {
				sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
				    m_local_decompression_interface[{item.get_global_id(0) + m_x, item.get_global_id(1) + m_y, 0}]};

				int32_t status = 0;
				int32_t count = 0;

				if(item.get_global_id(2) == 0) {
					int32_t combined_status = atomic_ref_count.load();

					auto [separate_status, separate_count] = separate_status_atomic(combined_status);

					do {
						auto [separate_status, separate_count] = separate_status_atomic(combined_status);

						if(separate_status == decompressing) {
							status = is_decompressed;
							count = separate_count;
							m_local_decompression_interface_local[0] = is_decompressed;
						} else {
							status = separate_status;
							count = separate_count;
						}
					} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
				}
			}
		} else {
			if(item.get_global_id(0) + m_x < item.get_global_range(0) && item.get_global_id(1) + m_y < item.get_global_range(1)) {
				sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
				    m_local_decompression_interface[{item.get_global_id(0) + m_x, item.get_global_id(1) + m_y, 0}]};

				int32_t status = 0;
				int32_t count = 0;

				if(item.get_global_id(2) == 0) {
					int32_t combined_status = atomic_ref_count.load();

					do {
						auto [separate_status, separate_count] = separate_status_atomic(combined_status);

						status = separate_status;
						count = separate_count;

						if(status == compressing) {
							do {
								int32_t combined_status = atomic_ref_count.load();

								auto [separate_status, separate_count] = separate_status_atomic(combined_status);

								status = separate_status;
								count = separate_count;
							} while(status == compressing);
						}

						count++;

						if(status == is_compressed) {
							status = is_decompressed;
							m_local_decompression_interface_local[0] = is_decompressed;
						}
					} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
				}
			}
		}
	}

	~local_accessor_compressor() {
		if constexpr(detail::is_producer_mode(AccessMode)) {
			if(m_item.get_global_id(0) + m_x < m_item.get_global_range(0) && m_item.get_global_id(1) + m_y < m_item.get_global_range(1)) {
				sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
				    m_local_decompression_interface[{m_item.get_global_id(0) + m_x, m_item.get_global_id(1) + m_y, 0}]};

				int32_t status = 0;
				int32_t count = 0;

				if(m_item.get_local_id(2) == 0) {
					int32_t combined_status = atomic_ref_count.load();

					do {
						auto [separate_status, separate_count] = separate_status_atomic(combined_status);

						status = separate_status;
						count = separate_count;

						count--;

						if(count == 0 && status == is_decompressed) {
							status = compressing;
							m_local_decompression_interface_local[0] = compressing;
						} else if(count > 0) {
							m_local_decompression_interface_local[0] = is_decompressed;
						}
					} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
				}
			}

			celerity::group_barrier(m_item.get_group());

			if(m_local_decompression_interface_local[0] == compressing) {
				m_compression.compress(m_item, m_data_point_available, m_local_tile, m_compression_offset_acc, m_compressed_data, m_work_items_per_tile,
				    m_points_per_tile, m_min, m_tile_size);
			}

			celerity::group_barrier(m_item.get_group());

			if(m_item.get_global_id(0) + m_x < m_item.get_global_range(0) && m_item.get_global_id(1) + m_y < m_item.get_global_range(1)) {
				sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
				    m_local_decompression_interface[{m_item.get_global_id(0) + m_x, m_item.get_global_id(1) + m_y, 0}]};

				if(m_item.get_global_id(2) == 0) {
					int32_t combined_status = atomic_ref_count.load();
					auto [separate_status, separate_count] = separate_status_atomic(combined_status);

					if(separate_status == compressing) {
						atomic_ref_count.store(0);
						m_local_decompression_interface_local[0] = is_compressed;
					}
				}
			}
		} else {
			if(m_item.get_global_id(0) + m_x < m_item.get_global_range(0) && m_item.get_global_id(1) + m_y < m_item.get_global_range(1)) {
				sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> atomic_ref_count{
				    m_local_decompression_interface[{m_item.get_global_id(0) + m_x, m_item.get_global_id(1) + m_y, 0}]};

				int status = 0;
				int count = 0;

				if(m_item.get_local_id(2) == 0) {
					int32_t combined_status = atomic_ref_count.load();

					do {
						auto [separate_status, separate_count] = separate_status_atomic(combined_status);

						status = separate_status;
						count = separate_count;

						count--;

						if(count == 0 && status == is_decompressed) {
							status = is_compressed;
							m_local_decompression_interface_local[0] = is_compressed;
						} else if(count > 0) {
							m_local_decompression_interface_local[0] = is_decompressed;
						}
					} while(!atomic_ref_count.compare_exchange_strong(combined_status, combine_status_atomic(status, count)));
				}
			}
		}
	}

	local_accessor_compressor& operator=(const local_accessor_compressor&) = delete;
	local_accessor_compressor& operator=(local_accessor_compressor&&) = delete;


	inline DataT& operator[](const id<Dim>& index) const {
		return m_local_tile[celerity::detail::get_linear_index({m_item.get_global_range(0), m_item.get_global_range(1), m_points_per_tile}, index)];
	}

	template <int D = Dim, std::enable_if_t<(D > 0), int> = 0>
	inline decltype(auto) operator[](const size_t dim0) const {
		if(m_item.get_global_id(2) >= m_points_per_tile) { return; }
		return subscript_compressed(m_local_tile, dim0, m_item, 0);
	}

	local_accessor_compressor(const local_accessor_compressor&) = default;
	local_accessor_compressor(local_accessor_compressor&&) = default;

  private:
	celerity::nd_item<Dim> m_item;
	const Compression& m_compression;
	DataAvailable& m_data_point_available;
	UncompressedData m_local_tile;
	const GlobalOffset& m_compression_offset_acc;
	CompressedData& m_compressed_data;
	const int m_work_items_per_tile;
	const int m_points_per_tile;
	const int m_x;
	const int m_y;
	const DataT m_min;
	const int m_tile_size;
	DecompressionInterface& m_local_decompression_interface;
	LocalDecompressionInterface& m_local_decompression_interface_local;

	// bit manipulation for status and count combined atomic
	static constexpr int32_t bit_mask = 0b11;
	static constexpr int32_t shift = 2;

	static constexpr int is_compressed = 0;
	static constexpr int decompressing = 1;
	static constexpr int is_decompressed = 2;
	static constexpr int compressing = 3;

	std::pair<int32_t, int32_t> separate_status_atomic(int32_t status_atomic) const { return {status_atomic & bit_mask, status_atomic >> shift}; }

	int32_t combine_status_atomic(int32_t status, int32_t count) const { return (count << shift) | (status & bit_mask); }
};

template <typename DataT, int Dims, typename Intype>
class buffer<Intype, Dims, compressed<celerity::compression::point_cloud<Intype, DataT>>> : public buffer<DataT, Dims, compression::uncompressed> {
  public:
	using base = buffer<DataT, Dims, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;

	buffer(const Intype* data, range<Dims> range, compressed<compression>& compression)
	    : buffer(std::move(compression.compress_data(data, range.size())), range, compression), global_index({range.get(0), range.get(1), 1}),
	      device_local_buffer(range.get(2) * range.get(0) * range.get(1)), decompression_interface({range.get(0), range.get(1), 1}) {
		celerity::debug::set_buffer_name(global_index, "global_index");
		device_local_buffer.fill(0);
		decompression_interface.fill(0);
	}

	buffer(range<Dims> range, compressed<compression>& compression)
	    : base(range), m_compression(compression), global_index({range.get(0), range.get(1), 1}),
	      device_local_buffer(range.get(2) * range.get(0) * range.get(1)), decompression_interface({range.get(0), range.get(1), 1}) {
		celerity::debug::set_buffer_name(global_index, "global_index");
		device_local_buffer.fill(0);
		decompression_interface.fill(0);
	}

	const compressed<compression>& get_compression() const { return m_compression; }

	celerity::buffer<Intype, Dims> global_index;
	celerity::scratch_buffer<Intype, 1> device_local_buffer;
	celerity::scratch_buffer<int32_t, 3> decompression_interface;

  private:
	buffer(std::vector<DataT>&& data, range<Dims> range, compressed<compression>& compression)
	    : base(data.data(), range), m_data(std::move(data)), m_compression(compression), global_index({range[0], range[1]}) {}

	std::vector<DataT> m_data;

	compressed<compression> m_compression;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::device, compressed<celerity::compression::point_cloud<Intype, DataT>>>
    : public accessor<DataT, Dims, Mode, target::device, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::device, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;
	using quant_type = typename compression::compression_type;
	using value_type = typename compression::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, target::device> tag)
	    : base(buff, cgh, rmfn, tag), m_global_offset(buff.global_index, cgh, rmfn, tag), m_compression(buff.get_compression()),
	      m_points_per_tile(buff.get_range().get(2)), m_local_accessor(buff.device_local_buffer), m_local_decompression_interface(buff.decompression_interface),
	      m_local_decompression_interface_local({1}, cgh) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::device> tag,
	    const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_global_offset(buff.global_index, cgh, rmfn, tag, prop), m_compression(buff.get_compression()),
	      m_points_per_tile(buff.get_range().get(2)), m_local_accessor(buff.device_local_buffer), m_local_decompression_interface(buff.decompression_interface),
	      m_local_decompression_interface_local({1}, cgh) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::device> tag,
	    const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_global_offset(buff.global_index, cgh, celerity::access::all{}, tag, prop_list),
	      m_compression(buff.get_compression()), m_points_per_tile(buff.get_range().get(2)), m_local_accessor(buff.device_local_buffer),
	      m_local_decompression_interface(buff.decompression_interface), m_local_decompression_interface_local({1}, cgh) {}

	template <typename DataAccess>
	inline auto decompress_data(celerity::nd_item<3> item, int x, int y, DataAccess& data_point_available, const Intype min, const int tile_size) const {
		return local_accessor_compressor<Mode, Intype, 3, decltype(m_compression), decltype(m_global_offset), decltype(*this), decltype(m_local_accessor),
		    decltype(data_point_available), decltype(m_local_decompression_interface), decltype(m_local_decompression_interface_local)>(m_compression,
		    m_global_offset, *this, m_local_accessor, item, m_points_per_tile, item.get_local_range().get(2), data_point_available, x, y, min, tile_size,
		    m_local_decompression_interface, m_local_decompression_interface_local);
	}

	template <typename DataAccess>
	inline auto decompress_data(celerity::nd_item<3> item, DataAccess& data_point_available, const Intype min, const int tile_size) const {
		return decompress_data(item, 0, 0, data_point_available, min, tile_size);
	}

  private:
	celerity::accessor<Intype, Dims, Mode, target::device> m_global_offset;
	compressed<compression> m_compression;

	celerity::scratch_accessor<Intype, 1> m_local_accessor;
	mutable celerity::scratch_accessor<int32_t, 3> m_local_decompression_interface;
	mutable celerity::local_accessor<int32_t, 1> m_local_decompression_interface_local;

	const int m_points_per_tile;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode>
class accessor<DataT, Dims, Mode, target::host_task, compressed<celerity::compression::point_cloud<Intype, DataT>>>
    : public accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, target::host_task, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;
	using quant_type = typename compression::compression_type;
	using value_type = typename compression::value_type;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, target::host_task> tag)
	    : base(buff, cgh, rmfn, tag), m_global_offset(buff.global_index, cgh, rmfn, tag), m_compression(buff.get_compression()),
	      m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, target::host_task> tag,
	    const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_global_offset(buff.global_index, cgh, rmfn, tag, prop), m_compression(buff.get_compression()),
	      m_points_per_tile(buff.get_range().get(2)) {}

	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, target::host_task> tag,
	    const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_global_offset(buff.global_index, cgh, celerity::access::all{}, tag, prop_list),
	      m_compression(buff.get_compression()), m_points_per_tile(buff.get_range().get(2)) {}

	template <typename DataAccess>
	inline auto decompress_data(size_t width, size_t height, DataAccess& data_point_available) const {
		std::vector<std::vector<std::vector<Intype>>> uncompressed_data(
		    width, std::vector<std::vector<Intype>>(height, std::vector<Intype>(m_points_per_tile)));
		m_compression.decompress(*this, m_global_offset, data_point_available, uncompressed_data, width, height);
		return std::move(uncompressed_data);
	}

  private:
	celerity::accessor<Intype, Dims, Mode, target::host_task> m_global_offset;
	compressed<compression> m_compression;
	int m_points_per_tile;
};
} // namespace celerity