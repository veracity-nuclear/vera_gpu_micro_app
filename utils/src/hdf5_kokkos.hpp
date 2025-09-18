#pragma once

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>

static constexpr const char* SURF2CELL_LABEL = "surf2Cell";

// This function reads data from an HDF5 dataset and converts it to a specified Kokkos::View type.
template <typename ViewType>
ViewType HDF5ToKokkosView(const HighFive::DataSet &dataset, const std::string &label = "")
{
    static_assert(Kokkos::is_view<ViewType>::value, "ViewType must be a Kokkos::View type");

    // We need a view formatted correctly to read in the data from HDF5.
    using H5View = Kokkos::View<typename ViewType::data_type, Kokkos::HostSpace>;

    H5View h5View;
    ViewType d_view;

    std::vector<size_t> hdf5Extents = dataset.getDimensions();
    Kokkos::ViewAllocateWithoutInitializing d_alloc(label);
    Kokkos::ViewAllocateWithoutInitializing h5_alloc("H5 " + label);

    if constexpr (ViewType::rank == 0)
    {
        h5View = H5View(h5_alloc);
        d_view = ViewType(d_alloc);
    }
    else if constexpr (ViewType::rank == 1)
    {
        h5View = H5View(h5_alloc, hdf5Extents[0]);
        d_view = ViewType(d_alloc, hdf5Extents[0]);
    }
    else if constexpr (ViewType::rank == 2)
    {
        h5View = H5View(h5_alloc, hdf5Extents[0], hdf5Extents[1]);
        d_view = ViewType(d_alloc, hdf5Extents[0], hdf5Extents[1]);
    }
    else if constexpr (ViewType::rank == 3)
    {
        h5View = H5View(h5_alloc, hdf5Extents[0], hdf5Extents[1], hdf5Extents[2]);
        d_view = ViewType(d_alloc, hdf5Extents[0], hdf5Extents[1], hdf5Extents[2]);
    }
    else
    {
        static_assert(ViewType::rank <= 3, "Only up to 3D Kokkos::View is supported");
    }

    dataset.read(h5View.data());

    // Only adjust for surf2Cell if the view is 2D and integer type
    if constexpr (ViewType::rank == 2 && std::is_integral_v<typename ViewType::value_type>) {
        if (label == SURF2CELL_LABEL) {
            for (size_t i = 0; i < hdf5Extents[0]; ++i) {
                h5View(i, 0) -= 1; // Convert from 1-based to 0-based indexing
                h5View(i, 1) -= 1;
            }
        }
    }

    typename ViewType::HostMirror h_view = Kokkos::create_mirror_view(d_view);

    // These copies are no-op if the memory spaces are the same
    Kokkos::deep_copy(h_view, h5View); // Changes the layout to the host layout
    Kokkos::deep_copy(d_view, h_view); // Copy to device view

    return d_view;
}