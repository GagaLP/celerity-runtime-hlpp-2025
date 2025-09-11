#pragma once

#include <celerity.h>

namespace celerity {
template <typename T, typename Q>
class compressed<celerity::compression::point_cloud<T, Q>> {
	using compression_type = typename celerity::compression::point_cloud<T, Q>::compression_type;
	using value_type = typename celerity::compression::point_cloud<T, Q>::value_type;

  public:
	compressed(const value_type min, const int tile_size) : m_min(min), m_tile_size(tile_size) {};

	template <typename Item>
	compression_type compress(const value_type p, const Item item) const {
		value_type center = {m_min.x() + m_tile_size * item[0], m_min.y() + m_tile_size * item[1], 0};
		auto tmp = p - center;
		compression_type compressed_point = {tmp.x(), tmp.y(), tmp.z()};
		return compressed_point;
	}

	template <typename Item>
	value_type decompress(const compression_type p, const Item item) const {
		value_type center = {m_min.x() + m_tile_size * item[0], m_min.y() + m_tile_size * item[1], 0};
		value_type decompressed_p;

		decompressed_p.x() = p.x();
		decompressed_p.y() = p.y();
		decompressed_p.z() = p.z();

		decompressed_p += center;
		return decompressed_p;
	}

  private:
	const value_type m_min;
	const int m_tile_size;
};

template <typename T, typename C, int Dims, typename SelectedCompression>
struct uncompressed_item_wrapper_const {
  public:
	uncompressed_item_wrapper_const(const C& compressed_ref, const id<Dims>& item, const compressed<SelectedCompression>& compression)
	    : m_compressed_ref(compressed_ref), m_item(item), m_compression(compression) {}

	operator T() const { return m_compression.decompress(m_compressed_ref, m_item); }

  private:
	const C& m_compressed_ref;
	const id<Dims>& m_item;
	const compressed<SelectedCompression>& m_compression;
};

template <typename T, typename C, int Dims, typename SelectedCompression>
struct uncompressed_item_wrapper {
  public:
	uncompressed_item_wrapper(C& compressed_ref, const id<Dims>& item, const compressed<SelectedCompression>& compression)
	    : m_compressed_ref(compressed_ref), m_item(item), m_compression(compression) {}

	uncompressed_item_wrapper& operator=(T value) {
		m_compressed_ref = m_compression.compress(value, m_item);
		return *this;
	}

	operator T() const { return m_compression.decompress(m_compressed_ref, m_item); }
	explicit operator C() const { return m_compressed_ref; }

  private:
	C& m_compressed_ref;
	const id<Dims>& m_item;
	const compressed<SelectedCompression>& m_compression;
};

template <typename DataT, int Dims, typename Intype, access_mode Mode, target Target>
class accessor<DataT, Dims, Mode, Target, compressed<celerity::compression::point_cloud<Intype, DataT>>>
    : public accessor<DataT, Dims, Mode, Target, compression::uncompressed> {
  public:
	using base = accessor<DataT, Dims, Mode, Target, compression::uncompressed>;
	using compression = celerity::compression::point_cloud<Intype, DataT>;
	using compressed_type = typename compression::compression_type;
	using value_type = typename compression::value_type;
	using retval = std::conditional_t<detail::is_producer_mode(Mode), uncompressed_item_wrapper<Intype, DataT, Dims, compression>,
	    const uncompressed_item_wrapper_const<Intype, DataT, Dims, compression>>;

	template <typename T, int D, typename Functor, access_mode ModeNoInit>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<Mode, ModeNoInit, Target> tag)
	    : base(buff, cgh, rmfn, tag), m_compression(buff.get_compression()) {}

	template <typename T, int D, typename Functor, access_mode TagMode>
	accessor(buffer<T, D, compressed<compression>>& buff, handler& cgh, const Functor& rmfn, const detail::access_tag<TagMode, Mode, Target> tag,
	    const property::no_init& prop)
	    : base(buff, cgh, rmfn, tag, prop), m_compression(buff.get_compression()) {}


	template <typename T, int D, access_mode TagMode, access_mode TagModeNoInit>
	accessor(buffer<DataT, Dims, compressed<compression>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, Target> tag,
	    const property_list& prop_list)
	    : base(buff, cgh, access::all(), tag, prop_list), m_compression(buff.get_compression()) {}

	template <access_mode M = Mode>
	inline retval operator[](const id<Dims>& index) const {
		return { base::operator[](index), index, m_compression };
	}

	template <target T = Target, std::enable_if_t<T == target::host_task, int> = 0>
	inline std::vector<std::vector<std::vector<value_type>>> get_uncompressed_vec(range<Dims> new_range) const {
		std::vector<std::vector<std::vector<value_type>>> uncompressed_data(
		    new_range.get(0), std::vector<std::vector<value_type>>(new_range.get(1), std::vector<value_type>(new_range.get(2))));
		for(int i = 0; i < new_range.get(0); ++i) {
			for(int j = 0; j < new_range.get(1); ++j) {
				for(int k = 0; k < new_range.get(2); ++k) {
					uncompressed_data[i][j][k] =
					    m_compression.decompress(base::operator[]({static_cast<size_t>(i), static_cast<size_t>(j), static_cast<size_t>(k)}), id<3>(i, j, k));
				}
			}
		}

		return uncompressed_data;
	}

  private:
	compressed<compression> m_compression;
};


} // namespace celerity
