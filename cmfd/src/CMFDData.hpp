#pragma once

#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include "highfive/H5File.hpp"

template <typename T>
struct isKokkosDualView : std::false_type {};

template <typename DataType, typename... Properties>
struct isKokkosDualView<Kokkos::DualView<DataType, Properties...>> : std::true_type {};

template <typename DualViewType>
DualViewType HDF5ToKokkosView(const HighFive::DataSet &dataset, const std::string& label="")
{
    static_assert(isKokkosDualView<DualViewType>::value, "DualViewType must be a Kokkos::DualView type");
    // We need a view formatted correctly to read in the data from HDF5.
    using H5View = Kokkos::View<typename DualViewType::data_type, Kokkos::HostSpace>;

    DualViewType dualView;
    H5View h5View;

    std::vector<size_t> hdf5Extents = dataset.getDimensions();
    Kokkos::ViewAllocateWithoutInitializing dv_alloc("DV " + label);
    Kokkos::ViewAllocateWithoutInitializing h5_alloc("H5 " + label);

    if constexpr(DualViewType::rank == 0) {
        h5View = H5View(h5_alloc);
        dualView = DualViewType(dv_alloc);
    } else if constexpr(DualViewType::rank == 1) {
        h5View = H5View(h5_alloc, hdf5Extents[0]);
        dualView = DualViewType(dv_alloc, hdf5Extents[0]);
    } else if constexpr(DualViewType::rank == 2) {
        h5View = H5View(h5_alloc, hdf5Extents[0], hdf5Extents[1]);
        dualView = DualViewType(dv_alloc, hdf5Extents[0], hdf5Extents[1]);
    } else if constexpr(DualViewType::rank == 3) {
        h5View = H5View(h5_alloc, hdf5Extents[0], hdf5Extents[1], hdf5Extents[2]);
        dualView = DualViewType(dv_alloc, hdf5Extents[0], hdf5Extents[1], hdf5Extents[2]);
    } else {
        static_assert(DualViewType::rank <= 3, "Only up to 3D Kokkos::DualView is supported");
    }

    dataset.read(h5View.data());

    // DualViewType::t_host h_view = dualView.view_host();
    Kokkos::deep_copy(dualView.view_host(), h5View);
    dualView.template modify<typename DualViewType::host_mirror_space>();
    dualView.template sync<typename DualViewType::execution_space>();

    return dualView;
}