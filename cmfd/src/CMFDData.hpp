#pragma once

#include "Kokkos_Core.hpp"
#include "highfive/H5File.hpp"

template <typename ViewType>
ViewType HDF5ToKokkosView(const HighFive::DataSet &dataset, const std::string& label="")
{
    static_assert(Kokkos::is_view<ViewType>::value, "ViewType must be a Kokkos::View type");

    // We need a view formatted correctly to read in the data from HDF5.
    using H5View = Kokkos::View<typename ViewType::data_type, Kokkos::HostSpace>;

    H5View h5View;
    ViewType d_view;

    std::vector<size_t> hdf5Extents = dataset.getDimensions();
    Kokkos::ViewAllocateWithoutInitializing d_alloc(label);
    Kokkos::ViewAllocateWithoutInitializing h5_alloc("H5 " + label);

    if constexpr(ViewType::rank == 0) {
        h5View = H5View(h5_alloc);
        d_view = ViewType(d_alloc);
    } else if constexpr(ViewType::rank == 1) {
        h5View = H5View(h5_alloc, hdf5Extents[0]);
        d_view = ViewType(d_alloc, hdf5Extents[0]);
    } else if constexpr(ViewType::rank == 2) {
        h5View = H5View(h5_alloc, hdf5Extents[0], hdf5Extents[1]);
        d_view = ViewType(d_alloc, hdf5Extents[0], hdf5Extents[1]);
    } else if constexpr(ViewType::rank == 3) {
        h5View = H5View(h5_alloc, hdf5Extents[0], hdf5Extents[1], hdf5Extents[2]);
        d_view = ViewType(d_alloc, hdf5Extents[0], hdf5Extents[1], hdf5Extents[2]);
    } else {
        static_assert(ViewType::rank <= 3, "Only up to 3D Kokkos::View is supported");
    }

    dataset.read(h5View.data());
    typename ViewType::HostMirror h_view = Kokkos::create_mirror_view(d_view);

    // These copies are no-op if the memory spaces are the same
    Kokkos::deep_copy(h_view, h5View); // Changes the layout to the host layout
    Kokkos::deep_copy(d_view, h_view); // Copy to device view

    return d_view;
}

template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct CMFDData {
    static_assert(Kokkos::is_memory_space<MemorySpace>::value,
        "MemorySpace must be a Kokkos memory space");

    using View1D = Kokkos::View<PetscScalar *, MemorySpace>;
    using View2D = Kokkos::View<PetscScalar **, MemorySpace>;
    using ViewSurf2Cell = Kokkos::View<PetscInt *[2], MemorySpace>;
    using View3D = Kokkos::View<PetscScalar ***, MemorySpace>;

    size_t nCells, nSurfaces, nGroups;

    View1D volume;
    View2D chi;
    View2D dHat;
    View2D dTilde;
    View2D nuFissionXS;
    View2D pastFlux;
    View2D removalXS;
    ViewSurf2Cell surf2Cell;
    View3D scatteringXS;

    CMFDData() = default;

    CMFDData(const HighFive::Group& CMFDCoarseMesh){
        volume = HDF5ToKokkosView<View1D>(CMFDCoarseMesh.getDataSet("volume"), "volume");
        chi = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("chi"), "chi");
        dHat = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("Dhat"), "Dhat");
        dTilde = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("Dtilde"), "Dtilde");
        nuFissionXS = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("nu-fission XS"), "nuFissionXS");
        pastFlux = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("flux"), "pastFlux");
        removalXS = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("removal XS"), "removalXS");
        surf2Cell = HDF5ToKokkosView<ViewSurf2Cell>(CMFDCoarseMesh.getDataSet("surf2cell"), "surf2Cell");
        scatteringXS = HDF5ToKokkosView<View3D>(CMFDCoarseMesh.getDataSet("scattering XS"), "scatteringXS");

        size_t firstCell, lastCell;
        CMFDCoarseMesh.getDataSet("first cell").read(firstCell);
        CMFDCoarseMesh.getDataSet("last cell").read(lastCell);
        nCells = lastCell - firstCell + 1;

        nSurfaces = surf2Cell.extent(0);

        CMFDCoarseMesh.getDataSet("energy groups").read(nGroups);
    };
};