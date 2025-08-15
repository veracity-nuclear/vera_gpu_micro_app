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

inline PetscInt assignCellSurface(
    const std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> &cellToSurf,
    const Kokkos::View<PetscInt *, Kokkos::LayoutStride, Kokkos::HostSpace> &surfToOtherCell,
    const PetscInt thisCell,
    const PetscInt otherCell)
{
    /*
      Determines the position of the surface that should be replaced in cellToSurf[thisCell] when there are more
      than three surfaces for a cell in a direction (positive or negative). Returns the index of the surface to
      replace (0, 1, or 2), or -1 if no replacement is needed.

      |------------------|
      |                  |
      |surf1,2,3otherCell|
      |                  |
      |----surf1,2,3-----|
      |                  |
      |   this cell      |
      |                  |
      |---trial surf-----|
      |                  |
      |   other cell     |
      |                  | trial surf/other cell could be surf/otherCell123
      |------------------|

      If the cellToSurf map we are editing is for positive surfaces (i.e., surfaces where the corresponding cell is the
      positive cell (0th column in the surf2Cell map)), then thisCell is positive and otherCell is negative. The
      surfToOtherCell map should be surf2NegCell.
     */
    const PetscInt surf1 = cellToSurf[thisCell][0];
    const PetscInt surf2 = cellToSurf[thisCell][1];
    const PetscInt surf3 = cellToSurf[thisCell][2];

    const PetscInt surf1OtherCell = surfToOtherCell(surf1);
    const PetscInt surf2OtherCell = surfToOtherCell(surf2);
    const PetscInt surf3OtherCell = surfToOtherCell(surf3);

    // If two surfaces in the existing list for thisCell are the same, we will replace the surface with the higher index.
    // (Same means they have the same "other" cell since it is guaranteed that the "this" cell is the same)
    if (surf1OtherCell == surf2OtherCell && surf1OtherCell != -1)
        return (surf1 > surf2) ? 0 : 1;
    if (surf1OtherCell == surf3OtherCell && surf1OtherCell != -1)
        return (surf1 > surf3) ? 0 : 2;
    if (surf2OtherCell == surf3OtherCell && surf2OtherCell != -1)
        return (surf2 > surf3) ? 1 : 2;

    // If the trial surface is the same as a surface in the list, we keep the surface with the lower index.
    if (otherCell == surf1OtherCell)
        return (surf1OtherCell < otherCell) ? 0 : -2;
    if (otherCell == surf2OtherCell)
        return (surf2OtherCell < otherCell) ? 1 : -2;
    if (otherCell == surf3OtherCell)
        return (surf3OtherCell < otherCell) ? 2 : -2;

    // If none of the above are true, we have four surfaces for this cell that each have a different "other" cell.
    throw std::runtime_error("More than 3 unique surfaces found for a single cell, which is unexpected.");
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
    //  Negative numbers means no surface. We expect up to three surfaces per cell (3D cartesian).
    // Uses members surf2Cell and nCells.
    void buildCellToSurfsMapping()
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cell2PosSurfData(nCells, {-1, -1, -1});
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cell2NegSurfData(nCells, {-1, -1, -1});

        size_t nSurfaces = surf2Cell.extent(0);
        auto h_surf2PosCell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(surf2Cell, Kokkos::ALL(), 0));
        auto h_surf2NegCell = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(surf2Cell, Kokkos::ALL(), 1));

        // We need a separate vector to hold the surfaces where the exterior (-1) cell is positive
        // since we don't know how many there will be. Overestimate on the reserved size.
        std::vector<PetscInt> vecPosLeakageSurfs;
        vecPosLeakageSurfs.reserve(nSurfaces);

        // Count array to track how many surfaces we've found for each cell
        std::vector<PetscInt> countPosSurfPerCell(nCells, 0);
        std::vector<PetscInt> countNegSurfPerCell(nCells, 0);

        for (size_t surfID = 0; surfID < nSurfaces; surfID++)
        {
            const PetscInt posCell = h_surf2PosCell(surfID);
            const PetscInt negCell = h_surf2NegCell(surfID);

            if (posCell >= 0)
            {
                if (countPosSurfPerCell[posCell] < MAX_POS_SURF_PER_CELL)
                {
                    cell2PosSurfData[posCell][countPosSurfPerCell[posCell]] = surfID;
                    countPosSurfPerCell[posCell]++;
                }
                else
                {
                    PetscInt surfPosition = assignCellSurface(cell2PosSurfData, h_surf2NegCell, posCell, negCell);
                    if (surfPosition >= 0)
                    {

                        cell2PosSurfData[posCell][surfPosition] = surfID;
                    }
                }
            }
            else {
                vecPosLeakageSurfs.push_back(surfID);
            }

            if (negCell >= 0)
            {
                if (countNegSurfPerCell[negCell] < MAX_POS_SURF_PER_CELL)
                {
                    cell2NegSurfData[negCell][countNegSurfPerCell[negCell]] = surfID;
                    countNegSurfPerCell[negCell]++;
                }
                else
                {
                    PetscInt surfPosition = assignCellSurface(cell2NegSurfData, h_surf2PosCell, negCell, posCell);
                    if (surfPosition >= 0)
                    {
                        cell2NegSurfData[negCell][surfPosition] = surfID;
                    }
                }
            }
        }

        // Copy to the device view stored in the struct
        cell2PosSurf = ViewCellToSurfs("cell2PosSurf", nCells, MAX_POS_SURF_PER_CELL);
        cell2NegSurf = ViewCellToSurfs("cell2NegSurf", nCells, MAX_POS_SURF_PER_CELL);
        auto h_cell2PosSurf = Kokkos::create_mirror_view(cell2PosSurf);
        auto h_cell2NegSurf = Kokkos::create_mirror_view(cell2NegSurf);
        for (size_t i = 0; i < nCells; i++)
        {
            for (size_t j = 0; j < MAX_POS_SURF_PER_CELL; j++)
            {
                h_cell2PosSurf(i, j) = cell2PosSurfData[i][j];
                h_cell2NegSurf(i, j) = cell2NegSurfData[i][j];
            }
        }
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