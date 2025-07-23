// Defines a class and method for reading data templated on a memory space.
// Templates don't play well with headers AND bodies, so we just use a header.
#pragma once

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>
#include <petscsys.h>

constexpr size_t MAX_POS_SURF_PER_CELL = 3;
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

// This struct is used to store the data for the CMFD coarse mesh.
template <typename AssemblySpace = Kokkos::DefaultExecutionSpace>
struct CMFDData
{
    using MemorySpace = typename AssemblySpace::memory_space;
    static_assert(Kokkos::is_memory_space<MemorySpace>::value,
                  "MemorySpace must be a Kokkos memory space");

    using View1D = Kokkos::View<PetscScalar *, MemorySpace>;
    using View2D = Kokkos::View<PetscScalar **, MemorySpace>;
    using ViewSurfToCell = Kokkos::View<PetscInt *[2], MemorySpace>;
    using ViewCellToSurfs = Kokkos::View<PetscInt *[MAX_POS_SURF_PER_CELL], MemorySpace>;
    using View3D = Kokkos::View<PetscScalar ***, MemorySpace>;

    size_t nCells{}, nSurfaces{}, nGroups{}, nPosLeakageSurfs{};

    View1D volume;
    View2D chi;
    View2D dHat;
    View2D dTilde;
    View2D nuFissionXS;
    View2D pastFlux;
    View2D removalXS;
    View2D transportXS;
    ViewSurfToCell surf2Cell;
    View3D scatteringXS;

    // Calculated by buildCellToSurfsMapping
    ViewCellToSurfs cell2PosSurf;
    ViewCellToSurfs cell2NegSurf;
    View1D posLeakageSurfs;

    CMFDData() = default;

    // Constructor that reads the data from the HDF5 file.
    CMFDData(const HighFive::Group &CMFDCoarseMesh)
    {
        volume = HDF5ToKokkosView<View1D>(CMFDCoarseMesh.getDataSet("volume"), "volume");
        chi = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("chi"), "chi");
        dHat = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("Dhat"), "Dhat");
        dTilde = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("Dtilde"), "Dtilde");
        nuFissionXS = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("nu-fission XS"), "nuFissionXS");
        pastFlux = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("flux"), "pastFlux");
        removalXS = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("removal XS"), "removalXS");
        transportXS = HDF5ToKokkosView<View2D>(CMFDCoarseMesh.getDataSet("transport XS"), "transportXS");
        scatteringXS = HDF5ToKokkosView<View3D>(CMFDCoarseMesh.getDataSet("scattering XS"), "scatteringXS");

        // Based on the label "surf2Cell", HDF5ToKokkosView will convert the 1-based indexing to 0-based indexing.
        surf2Cell = HDF5ToKokkosView<ViewSurfToCell>(CMFDCoarseMesh.getDataSet("surf2cell"), SURF2CELL_LABEL);
        nSurfaces = surf2Cell.extent(0);

        size_t firstCell, lastCell;
        CMFDCoarseMesh.getDataSet("first cell").read(firstCell);
        CMFDCoarseMesh.getDataSet("last cell").read(lastCell);
        nCells = lastCell - firstCell + 1;

        CMFDCoarseMesh.getDataSet("energy groups").read(nGroups);

        buildCellToSurfsMapping();
    };

    // Build a mapping from a cell to the surfaces in which the cell is positive (north, up, right).
    // Therefore the "pos" surf would be negative (south, down, left) for the cell. See test for an example.
    // -1 Means no surface. We expect up to three surfaces per cell (3D cartesian).
    // Uses members surf2Cell and nCells.
    void buildCellToSurfsMapping()
    {
        cell2PosSurf = ViewCellToSurfs("cell2PosSurf", nCells, MAX_POS_SURF_PER_CELL);
        auto h_cell2PosSurf = Kokkos::create_mirror_view(cell2PosSurf);
        Kokkos::deep_copy(h_cell2PosSurf, -1);

        cell2NegSurf = ViewCellToSurfs("cell2NegSurf", nCells, MAX_POS_SURF_PER_CELL);
        auto h_cell2NegSurf = Kokkos::create_mirror_view(cell2NegSurf);
        Kokkos::deep_copy(h_cell2NegSurf, -1);

        auto h_surf2Cell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), surf2Cell);
        size_t nSurfaces = h_surf2Cell.extent(0);

        // We need a separate vector to hold the surfaces where the exterior (-1) cell is positive
        // since we don't know how many there will be. Overestimate on the reserved size.
        std::vector<PetscInt> vecPosLeakageSurfs;
        vecPosLeakageSurfs.reserve(nSurfaces);

        // Count array to track how many surfaces we've found for each cell
        std::vector<PetscInt> countPosSurfPerCell(nCells, 0);
        std::vector<PetscInt> countNegSurfPerCell(nCells, 0);

        for (size_t surfID = 0; surfID < nSurfaces; surfID++)
        {
            const PetscInt posCell = h_surf2Cell(surfID, 0);
            const PetscInt negCell = h_surf2Cell(surfID, 1);

            if (posCell >= 0)
            {
                if (countPosSurfPerCell[posCell] < MAX_POS_SURF_PER_CELL)
                {
                    h_cell2PosSurf(posCell, countPosSurfPerCell[posCell]) = surfID;
                    countPosSurfPerCell[posCell]++;
                }
                else
                {
                    throw std::runtime_error("More than 3 positive surfaces found for a single cell, which is unexpected.");
                }
            }
            else {
                vecPosLeakageSurfs.push_back(surfID);
            }

            if (negCell >= 0)
            {
                if (countNegSurfPerCell[negCell] < MAX_POS_SURF_PER_CELL)
                {
                    h_cell2NegSurf(negCell, countNegSurfPerCell[negCell]) = surfID;
                    countNegSurfPerCell[negCell]++;
                }
                else
                {
                    throw std::runtime_error("More than 3 negative surfaces found for a single cell, which is unexpected.");
                }
            }
        }

        // Copy to the device view stored in the struct
        Kokkos::deep_copy(cell2PosSurf, h_cell2PosSurf);
        Kokkos::deep_copy(cell2NegSurf, h_cell2NegSurf);

        // Create a view for the positive leakage surfaces and store in the struct
        nPosLeakageSurfs = vecPosLeakageSurfs.size();
        posLeakageSurfs = View1D("posLeakageSurfs", nPosLeakageSurfs);
        auto h_posLeakageSurfs = Kokkos::create_mirror_view(posLeakageSurfs);
        for (size_t i = 0; i < nPosLeakageSurfs; i++)
        {
            h_posLeakageSurfs(i) = vecPosLeakageSurfs[i];
        }
        Kokkos::deep_copy(posLeakageSurfs, h_posLeakageSurfs);
    }
};