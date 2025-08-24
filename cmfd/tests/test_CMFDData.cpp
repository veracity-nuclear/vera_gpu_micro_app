/*
  This file
*/
#include "PetscKokkosTestEnvironment.hpp"
#include "CMFDData.hpp"

TEST(readData, testHDF5ToKokkosView)
{
    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::DataSet H5Dataset = file.getDataSet("CMFD_CoarseMesh/flux");
    std::vector<size_t> dims = H5Dataset.getDimensions();

    std::vector<std::vector<double>> originalData(dims[0], std::vector<double>(dims[1], 0.0));
    H5Dataset.read(originalData);

    auto d_view = HDF5ToKokkosView<Kokkos::View<double**>>(H5Dataset, "flux");

    // Check if the data is correctly copied to the device
    Kokkos::View<double**>::HostMirror h_view = Kokkos::create_mirror_view(d_view);
    Kokkos::deep_copy(h_view, d_view);
    for (size_t i = 0; i < h_view.extent(0); ++i) {
        for (size_t j = 0; j < h_view.extent(1); ++j) {
            ASSERT_DOUBLE_EQ(h_view(i, j), originalData[i][j]) << "Error copying data from HDF5 to Device";
        }
    }
}

TEST(readData, initializeCoarseData)
{
    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    const HighFive::Group CMFDCoarseMesh = file.getGroup("CMFD_CoarseMesh");

    // Get data the standard way
    size_t firstCell, lastCell, nEnergyGroups;
    std::vector<PetscScalar> volume;
    std::vector<std::vector<PetscScalar>> chi, Dhat, Dtilde, nuFissionXs, removalXs, transportXs;
    std::vector<std::vector<PetscInt>> surf2Cell;
    std::vector<std::vector<std::vector<PetscScalar>>> scatteringXs;

    CMFDCoarseMesh.getDataSet("first cell").read(firstCell);
    CMFDCoarseMesh.getDataSet("last cell").read(lastCell);
    CMFDCoarseMesh.getDataSet("energy groups").read(nEnergyGroups);
    CMFDCoarseMesh.getDataSet("volume").read(volume);
    CMFDCoarseMesh.getDataSet("chi").read(chi);
    CMFDCoarseMesh.getDataSet("Dhat").read(Dhat);
    CMFDCoarseMesh.getDataSet("Dtilde").read(Dtilde);
    CMFDCoarseMesh.getDataSet("nu-fission XS").read(nuFissionXs);
    CMFDCoarseMesh.getDataSet("removal XS").read(removalXs);
    CMFDCoarseMesh.getDataSet("transport XS").read(transportXs);
    CMFDCoarseMesh.getDataSet("scattering XS").read(scatteringXs);
    CMFDCoarseMesh.getDataSet("surf2cell").read(surf2Cell);

    // convert surf2Cell from 1-based to 0-based indexing
    for (size_t i = 0; i < surf2Cell.size(); ++i) {
        surf2Cell[i][0] -= 1;
        surf2Cell[i][1] -= 1;
    }

    size_t nCells = lastCell - firstCell + 1;
    size_t nSurfaces = surf2Cell.size();

    // Get data using the h_data
    const CMFDData<Kokkos::HostSpace> h_data(CMFDCoarseMesh);

    ASSERT_EQ(h_data.nCells, nCells) << "Number of cells mismatch";
    ASSERT_EQ(h_data.nSurfaces, nSurfaces) << "Number of surfaces mismatch";
    ASSERT_EQ(h_data.nGroups, nEnergyGroups) << "Number of energy groups mismatch";

    compare2DViewAndVector(h_data.chi, chi, "Chi data mismatch");
    compare2DViewAndVector(h_data.dHat, Dhat, "Dhat data mismatch");
    compare2DViewAndVector(h_data.dTilde, Dtilde, "Dtilde data mismatch");
    compare2DViewAndVector(h_data.nuFissionXS, nuFissionXs, "Nu Fission XS data mismatch");
    compare2DViewAndVector(h_data.removalXS, removalXs, "Removal XS data mismatch");
    compare2DViewAndVector(h_data.transportXS, transportXs, "Transport XS data mismatch");

    auto h_volume = h_data.volume;
    for (size_t i = 0; i < nCells; ++i) {
        ASSERT_DOUBLE_EQ(h_volume(i), volume[i]) << "Volume data mismatch at index " << i;
    }

    auto h_surf2Cell = h_data.surf2Cell;
    for (size_t i = 0; i < nSurfaces; ++i) {
        ASSERT_EQ(h_surf2Cell(i, 0), surf2Cell[i][0]) << "Surf2Cell data mismatch at index " << i << ", first element";
        ASSERT_EQ(h_surf2Cell(i, 1), surf2Cell[i][1]) << "Surf2Cell data mismatch at index " << i << ", second element";
    }

    auto h_scatteringXs = h_data.scatteringXS;
    for (size_t eg1 = 0; eg1 < nEnergyGroups; ++eg1) {
        for (size_t eg2 = 0; eg2 < nEnergyGroups; ++eg2) {
            for (size_t i = 0; i < nCells; ++i) {
                ASSERT_DOUBLE_EQ(h_scatteringXs(eg1, eg2, i), scatteringXs[eg1][eg2][i])
                    << "Scattering XS data mismatch at (" << eg1 << ", " << eg2 << ", " << i << ")";
            }
        }
    }

    // Compare the data on the device
    const CMFDData<> d_data(CMFDCoarseMesh);

    compare2DHostAndDevice(h_data.chi, d_data.chi, "Chi data mismatch on device");
    compare2DHostAndDevice(h_data.dHat, d_data.dHat, "Dhat data mismatch on device");
    compare2DHostAndDevice(h_data.dTilde, d_data.dTilde, "Dtilde data mismatch on device");
    compare2DHostAndDevice(h_data.nuFissionXS, d_data.nuFissionXS, "Nu Fission XS data mismatch on device");
    compare2DHostAndDevice(h_data.removalXS, d_data.removalXS, "Removal XS data mismatch on device");
    compare2DHostAndDevice(h_data.transportXS, d_data.transportXS, "Transport XS data mismatch on device");

    auto h_volumeCheck = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_data.volume);
    for (size_t i = 0; i < nCells; ++i) {
        ASSERT_DOUBLE_EQ(h_volumeCheck(i), volume[i]) << "Volume data mismatch on device at index " << i;
    }

    auto h_surf2CellCheck = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_data.surf2Cell);
    for (size_t i = 0; i < nSurfaces; ++i) {
        ASSERT_EQ(h_surf2CellCheck(i, 0), surf2Cell[i][0]) << "Surf2Cell data mismatch on device at index " << i << ", first element";
        ASSERT_EQ(h_surf2CellCheck(i, 1), surf2Cell[i][1]) << "Surf2Cell data mismatch on device at index " << i << ", second element";
    }

    auto h_scatteringXsCheck = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_data.scatteringXS);
    for (size_t eg1 = 0; eg1 < nEnergyGroups; ++eg1) {
        for (size_t eg2 = 0; eg2 < nEnergyGroups; ++eg2) {
            for (size_t i = 0; i < nCells; ++i) {
                ASSERT_DOUBLE_EQ(h_scatteringXsCheck(eg1, eg2, i), scatteringXs[eg1][eg2][i])
                    << "Scattering XS data mismatch on device at (" << eg1 << ", " << eg2 << ", " << i << ")";
            }
        }
    }
}

TEST(surf2CellToCell2Surf, test)
{
    /*
    cell -1 is outside the mesh
    Positive surfaces are up or to the right

    +-----21----+-----22----+-----23----+
    |           |           |           |
    |     6     |     7     |     8     |
  17|         18|         19|           | 20
    |           |           |           |
    +-----14----+-----15----+-----16----+
    |           |           |           |
    |     3     |     4     |     5     |
  10|         11|         12|           | 13
    |           |           |           |
    +------7----+------8----+------9----+
    |           |           |           |
    |     0     |     1     |     2     |
   3|          4|          5|           | 6
    |           |           |           |
    +------0----+------1----+------2----+
    */
    CMFDData<> cmfdData;
    cmfdData.nCells = 9;
    cmfdData.nSurfaces = 24;

    // index is surface number (0 based)
    std::vector<std::array<PetscInt, 2>> surf2Cell = {
        {0,-1}, {1,-1}, {2,-1}, // Surf0: +cell0 -cell-1, Surf1: +cell1 -cell-1, Surf2: +cell2 -cell-1
        {0,-1}, {1,0}, {2,1}, {-1,2},
        {3,0}, {4,1}, {5,2}, // Surf7: +cell3 -cell0
        {3,-1}, {4,3}, {5,4}, {-1,5},
        {6,3}, {7,4}, {8,5},
        {6,-1}, {7,6}, {8,7}, {-1,8},
        {-1,6}, {-1,7}, {-1,8}
    };

    // index is cell number (0 based)
    std::vector<std::array<PetscInt, 3>> expectedCellToPosSurf = {
        {0, 3, -1}, // Cell 0: Surf0 and Surf3 (and -1 since 2D)
        {1, 4, -1}, // Cell 1: Surf1 and Surf4
        {2, 5, -1}, // Cell 2
        {7, 10, -1},
        {8, 11, -1},
        {9, 12, -1},
        {14, 17, -1},
        {15, 18, -1},
        {16, 19, -1},
    };

    // Surfaces in which the exterior (-1) cell is positive
    std::vector<PetscInt> posLeakageSurfs = {6, 13, 20, 21, 22, 23};

    CMFDData<>::ViewSurfToCell d_surf2CellView("surf2Cell", cmfdData.nSurfaces, 2);
    auto h_surf2CellView = Kokkos::create_mirror_view(d_surf2CellView);
    for (size_t i = 0; i < cmfdData.nSurfaces; ++i) {
        h_surf2CellView(i, 0) = surf2Cell[i][0];
        h_surf2CellView(i, 1) = surf2Cell[i][1];
    }
    Kokkos::deep_copy(d_surf2CellView, h_surf2CellView);

    cmfdData.surf2Cell = d_surf2CellView;
    cmfdData.buildCellToSurfsMapping();

    auto h_cell2PosSurf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cmfdData.cell2PosSurf);
    for (size_t i = 0; i < cmfdData.nCells; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(h_cell2PosSurf(i, j), expectedCellToPosSurf[i][j])
                << "Mismatch at cell " << i << ", position " << j;
            // printf("Cell %zu, Position %zu: %d (Expected: %d)\n", i + 1, j + 1, h_cell2PosSurf(i, j), expectedCellToPosSurf[i][j]);
        }
    }

    EXPECT_EQ(cmfdData.nPosLeakageSurfs, posLeakageSurfs.size()) << "Number of positive leakage surfaces mismatch";

    auto h_posLeakageSurfs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cmfdData.posLeakageSurfs);
    for (size_t i = 0; i < posLeakageSurfs.size(); ++i) {
        EXPECT_EQ(h_posLeakageSurfs(i), posLeakageSurfs[i])
            << "Mismatch at position " << i << ": " << h_posLeakageSurfs(i) << " (Expected: " << posLeakageSurfs[i] << ")";
    }
}

TEST(assignCellSurface, basicLogic)
{
    // This test checks the logic of assignCellSurface function
    // An LLM generated these...

    // Setup: 3 surfaces per cell, each with a different "other" cell
    std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = {
        {0, 1, 2}
    };
    // surfToOtherCell: surface 0 -> 10, 1 -> 11, 2 -> 12
    Kokkos::View<PetscInt*, Kokkos::HostSpace> surfToOtherCell("surfToOtherCell", 3);
    surfToOtherCell(0) = 10;
    surfToOtherCell(1) = 11;
    surfToOtherCell(2) = 12;

    // No duplicates, no replacement needed
    EXPECT_THROW(assignCellSurface(cellToSurf, surfToOtherCell, 0, 13), std::runtime_error);

    // Duplicate "other" cell for surfaces 0 and 1
    surfToOtherCell(1) = 10;
    // surf0(0) vs surf1(1), surf1 > surf0, so replace 1
    EXPECT_EQ(assignCellSurface(cellToSurf, surfToOtherCell, 0, 13), 1);

    // Duplicate "other" cell for surfaces 0 and 2
    surfToOtherCell(1) = 11;
    surfToOtherCell(2) = 10;
    // surf2(2) > surf0(0), so replace 2
    EXPECT_EQ(assignCellSurface(cellToSurf, surfToOtherCell, 0, 13), 2);

    // Duplicate "other" cell for surfaces 1 and 2
    surfToOtherCell(0) = 10;
    surfToOtherCell(1) = 12;
    surfToOtherCell(2) = 12;
    // surf2(2) > surf1(1), so replace 2
    EXPECT_EQ(assignCellSurface(cellToSurf, surfToOtherCell, 0, 13), 2);

    // Test trial surface matches an existing surface
    surfToOtherCell(0) = 10;
    surfToOtherCell(1) = 11;
    surfToOtherCell(2) = 12;
    // otherCell == surf1OtherCell
    EXPECT_EQ(assignCellSurface(cellToSurf, surfToOtherCell, 0, 10), -2);
    // otherCell == surf2OtherCell
    EXPECT_EQ(assignCellSurface(cellToSurf, surfToOtherCell, 0, 11), -2);
    // otherCell == surf3OtherCell
    EXPECT_EQ(assignCellSurface(cellToSurf, surfToOtherCell, 0, 12), -2);

        // Helper to build a view sized to (maxSurfId+1) and init to -999
    auto makeView = [](std::initializer_list<std::pair<int,int>> pairs) {
        int maxId = -1;
        for (auto &p : pairs) maxId = std::max(maxId, p.first);
        Kokkos::View<PetscInt*, Kokkos::HostSpace> v("surfToOtherCell", maxId + 1);
        for (int i = 0; i <= maxId; ++i) v(i) = -999;
        for (auto &p : pairs) v(p.first) = p.second;
        return v;
    };

    // 1. Three unique others -> adding a 4th unique should throw
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {0,1,2} };
        auto surfToOther = makeView({ {0,10},{1,11},{2,12} });
        EXPECT_THROW(assignCellSurface(cellToSurf, surfToOther, 0, 13), std::runtime_error);
    }

    // 2. Duplicate (surf0,surf1) with surf1 > surf0 -> expect index 1
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {0,1,2} };
        auto surfToOther = makeView({ {0,10},{1,10},{2,12} });
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 13), 1);
    }

    // 3. Duplicate (surf0,surf1) with surf0 > surf1 (reverse ordering) -> expect index 0
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {2,1,0} };
        auto surfToOther = makeView({ {2,55},{1,55},{0,70} });
        // Duplicates between surfaces 2 and 1 -> higher surface ID (2) replaced => index 0
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 99), 0);
    }

    // 4. Duplicate (surf0,surf2) path (surf2 > surf0) -> expect index 2
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {0,4,7} };
        auto surfToOther = makeView({ {0,20},{4,30},{7,20} });
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 88), 2);
    }

    // 5. Duplicate (surf0,surf2) with surf0 > surf2 (reverse ordering) -> expect index 0
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {9,3,4} };
        auto surfToOther = makeView({ {9,90},{3,15},{4,90} });
        // Duplicate between surfaces 9 and 4 -> higher surface ID 9 replaced => index 0
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 777), 0);
    }

    // 6. Duplicate (surf1,surf2) with surf2 > surf1 -> expect index 2
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {5,8,11} };
        auto surfToOther = makeView({ {5,10},{8,42},{11,42} });
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 1234), 2);
    }

    // 7. Duplicate (surf1,surf2) with surf1 > surf2 (reverse ordering) -> expect index 1
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {4,13,7} };
        auto surfToOther = makeView({ {4,1},{13,77},{7,77} });
        // Duplicate between 13 and 7 -> higher surface ID 13 replaced => index 1
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 555), 1);
    }

    // 8. All three share same other cell -> first duplicate branch triggers (surf0,surf1)
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {2,6,4} };
        auto surfToOther = makeView({ {2,111},{6,111},{4,111} });
        // Branch (surf1Other == surf2Other); higher surface ID (6) replaced => index 1
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 200), 1);
    }

    // 9. Trial otherCell equal to existing (each position) -> expect -2
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {0,1,2} };
        auto surfToOther = makeView({ {0,10},{1,11},{2,12} });
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 10), -2);
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 11), -2);
        EXPECT_EQ(assignCellSurface(cellToSurf, surfToOther, 0, 12), -2);
    }

    // 10. Sentinel (-1) values: duplicates with -1 ignored; adding new unique triggers throw
    {
        std::vector<std::array<PetscInt, MAX_POS_SURF_PER_CELL>> cellToSurf = { {0,1,2} };
        auto surfToOther = makeView({ {0,-1},{1,-1},{2,-1} });
        EXPECT_THROW(assignCellSurface(cellToSurf, surfToOther, 0, 99), std::runtime_error);
    }
}

TEST(fissionableCellToXSCell, basic)
{
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using IndexList = FineMeshData<AssemblySpace>::ViewIndexList;

    auto buildAndCheck = [](const std::vector<bool> &isFissionable, const std::vector<PetscInt> &truthFissionToXSCell)
    {
        IndexList xsCellToNthFissionable;
        initializeXSCellToNthFissionable<IndexList>(isFissionable, xsCellToNthFissionable);

        ASSERT_EQ(xsCellToNthFissionable.extent(0), truthFissionToXSCell.size());
        for (size_t i = 0; i < truthFissionToXSCell.size(); ++i)
            ASSERT_EQ(xsCellToNthFissionable(i), truthFissionToXSCell[i]);
    };

    buildAndCheck({0, 0, 0, 0}, {-1, -1, -1, -1});
    buildAndCheck({1, 1, 1, 1, 1}, {0, 1, 2, 3, 4});
    buildAndCheck({1, 0, 1, 0, 1, 0, 1}, {0, -1, 1, -1, 2, -1, 3});
    buildAndCheck({0, 1, 0, 0, 1, 0, 0, 0, 1}, {-1, 0, -1, -1, 1, -1, -1, -1, 2});
    buildAndCheck({0, 0, 1, 0, 1, 0, 0}, {-1, -1, 0, -1, 1, -1, -1});
    buildAndCheck({1, 1, 1, 1, 1, 0, 0, 0}, {0, 1, 2, 3, 4, -1, -1, -1});
    buildAndCheck({1}, {0});
    buildAndCheck({0}, {-1});
}

TEST(readData, initializeFineData)
{
    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group fineGroup = file.getGroup("CMFD_FineMesh");

    // Read (reference) data directly via HighFive
    std::vector<PetscInt> coarseToXSCells_vec, xsToFineCells_vec;
    std::vector<PetscScalar> volumePerXSR_vec;
    std::vector<std::vector<PetscScalar>> fineFlux_vec, transportXS_vec, nuFissionXS_vec, chi_vec;
    std::vector<std::vector<std::vector<PetscScalar>>> scatteringXS_vec;

    fineGroup.getDataSet("nxscells").read(coarseToXSCells_vec);
    fineGroup.getDataSet("nfinecells").read(xsToFineCells_vec);
    fineGroup.getDataSet("volume").read(volumePerXSR_vec);
    fineGroup.getDataSet("flux").read(fineFlux_vec);
    fineGroup.getDataSet("transport XS").read(transportXS_vec);
    fineGroup.getDataSet("nu-fission XS").read(nuFissionXS_vec);
    fineGroup.getDataSet("chi").read(chi_vec);
    fineGroup.getDataSet("scattering XS").read(scatteringXS_vec);

    // Construct FineMeshData (host + device)
    FineMeshData<Kokkos::DefaultHostExecutionSpace> h_fine(fineGroup);
    FineMeshData<> d_fine(fineGroup);

    // Helper lambdas
    auto compare1DScalar = [](auto view, const std::vector<PetscScalar> &ref, const char *msg) {
        ASSERT_EQ(view.extent(0), ref.size()) << msg << " extent mismatch";
        for (size_t i = 0; i < ref.size(); ++i)
            ASSERT_DOUBLE_EQ(view(i), ref[i]) << msg << " mismatch at " << i;
    };
    auto compare1DInt = [](auto view, const std::vector<PetscInt> &ref, const char *msg) {
        ASSERT_EQ(view.extent(0), ref.size()) << msg << " extent mismatch";
        for (size_t i = 0; i < ref.size(); ++i)
            ASSERT_EQ(view(i), ref[i]) << msg << " mismatch at " << i;
    };
    auto compare2D = [](auto view, const std::vector<std::vector<PetscScalar>> &ref, const char *msg) {
        ASSERT_EQ(view.extent(0), ref.size()) << msg << " outer extent mismatch";
        if (!ref.empty())
            ASSERT_EQ(view.extent(1), ref[0].size()) << msg << " inner extent mismatch";
        for (size_t i = 0; i < ref.size(); ++i)
            for (size_t j = 0; j < ref[i].size(); ++j)
                ASSERT_DOUBLE_EQ(view(i, j), ref[i][j]) << msg << " mismatch at (" << i << "," << j << ")";
    };
    auto compare3D = [](auto view, const std::vector<std::vector<std::vector<PetscScalar>>> &ref, const char *msg) {
        ASSERT_EQ(view.extent(0), ref.size()) << msg << " dim0 mismatch";
        if (!ref.empty()) ASSERT_EQ(view.extent(1), ref[0].size()) << msg << " dim1 mismatch";
        if (!ref.empty() && !ref[0].empty()) ASSERT_EQ(view.extent(2), ref[0][0].size()) << msg << " dim2 mismatch";
        for (size_t g1 = 0; g1 < ref.size(); ++g1)
            for (size_t g2 = 0; g2 < ref[g1].size(); ++g2)
                for (size_t k = 0; k < ref[g1][g2].size(); ++k)
                    ASSERT_DOUBLE_EQ(view(g1, g2, k), ref[g1][g2][k])
                        << msg << " mismatch at (" << g1 << "," << g2 << "," << k << ")";
    };

    // Host mirrors
    auto h_coarseToXSCells = h_fine.coarseToXSCells;
    auto h_xsToFineCells = h_fine.xsToFineCells;
    auto h_volumePerXSR = h_fine.volumePerXSR;
    auto h_fineFlux = h_fine.fineFlux;
    auto h_transportXS = h_fine.transportXS;
    auto h_nuFissionXS = h_fine.nuFissionXS;
    auto h_chi = h_fine.chi;
    auto h_scatteringXS = h_fine.scatteringXS;

    // Compare host data
    compare1DInt(h_coarseToXSCells, coarseToXSCells_vec, "coarseToXSCells");
    compare1DInt(h_xsToFineCells, xsToFineCells_vec, "xsToFineCells");
    compare1DScalar(h_volumePerXSR, volumePerXSR_vec, "volumePerXSR");
    compare2D(h_fineFlux, fineFlux_vec, "fineFlux");
    compare2D(h_transportXS, transportXS_vec, "transportXS");
    compare2D(h_nuFissionXS, nuFissionXS_vec, "nuFissionXS");
    compare2D(h_chi, chi_vec, "chi");
    compare3D(h_scatteringXS, scatteringXS_vec, "scatteringXS");

    // Device mirrors -> host copies
    auto d_coarseToXSCells_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.coarseToXSCells);
    auto d_xsToFineCells_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.xsToFineCells);
    auto d_volumePerXSR_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.volumePerXSR);
    auto d_fineFlux_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.fineFlux);
    auto d_transportXS_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.transportXS);
    auto d_nuFissionXS_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.nuFissionXS);
    auto d_chi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.chi);
    auto d_scatteringXS_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_fine.scatteringXS);

    // Compare device-loaded data
    compare1DInt(d_coarseToXSCells_h, coarseToXSCells_vec, "device coarseToXSCells");
    compare1DInt(d_xsToFineCells_h, xsToFineCells_vec, "device xsToFineCells");
    compare1DScalar(d_volumePerXSR_h, volumePerXSR_vec, "device volumePerXSR");
    compare2D(d_fineFlux_h, fineFlux_vec, "device fineFlux");
    compare2D(d_transportXS_h, transportXS_vec, "device transportXS");
    compare2D(d_nuFissionXS_h, nuFissionXS_vec, "device nuFissionXS");
    compare2D(d_chi_h, chi_vec, "device chi");
    compare3D(d_scatteringXS_h, scatteringXS_vec, "device scatteringXS");

    // Check scalar values
    HighFive::Group coarseMesh = file.getGroup("CMFD_CoarseMesh");
    PetscInt energyGroups, firstCell, lastCell;
    coarseMesh.getDataSet("energy groups").read(energyGroups);
    ASSERT_EQ(d_fine.nEnergyGroups, energyGroups) << "Energy groups mismatch";

    coarseMesh.getDataSet("first cell").read(firstCell);
    coarseMesh.getDataSet("last cell").read(lastCell);
    PetscInt nCoarseCells = lastCell - firstCell + 1;
    ASSERT_EQ(d_fine.nCoarseCells, nCoarseCells) << "Coarse cells mismatch";

    PetscInt nXSCells = coarseToXSCells_vec.back();
    ASSERT_EQ(d_fine.nXSCells, nXSCells) << "XS cells mismatch";

    PetscInt nFineCells = xsToFineCells_vec.back();
    ASSERT_EQ(d_fine.nFineCells, nFineCells) << "Fine cells mismatch";

    // Check if initializeFissionableCellToXSCell has been run
    ASSERT_EQ(h_fine.xsCellToNthFissionable.extent(0), nXSCells);
    ASSERT_EQ(d_fine.xsCellToNthFissionable.extent(0), nXSCells);

    // Find the number of fissionable XS cells according to xsCellToNthFissionable
    PetscInt nFissionable = -1;
    for (size_t i = nXSCells; i-- > 0; )
    {
        nFissionable = h_fine.xsCellToNthFissionable(i);
        // exit on first since xsCellToNthFissionable is in order
        if (nFissionable != -1)
        {
            nFissionable++;
            break;
        }
    }

    // Check  extents of chi and nuFission
    ASSERT_EQ(d_fine.chi.extent(0), nFissionable);
    ASSERT_EQ(d_fine.nuFissionXS.extent(0), nFissionable);
    ASSERT_EQ(d_fine.chi.extent(1), d_fine.nEnergyGroups);
    ASSERT_EQ(d_fine.nuFissionXS.extent(1), d_fine.nEnergyGroups);
}

// TODO: Put into a test fixture so we can test different files
TEST(homogenization, fineFlux)
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;

    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group coarseGroup = file.getGroup("CMFD_CoarseMesh");
    HighFive::Group fineGroup = file.getGroup("CMFD_FineMesh");

    CMFDData<AssemblySpace> coarseData(coarseGroup);
    FineMeshData<AssemblySpace> fineData(fineGroup);

    FineMeshData<AssemblySpace>::View2D calculatedCoarseFlux = fineData.homogenizeFineFlux();

    // TODO: We have the relative and absolute tolerances set pretty high.
    // Aaron and I (Braden) looked at the coarse and fine outputs generated by MPACT and
    // could not find a great reason for the discrepancy. The data from the file are converged
    // to precision further than the tolerances we are using.
    compare2DViews<PetscScalar, AssemblySpace>(calculatedCoarseFlux, coarseData.pastFlux, 0.002, 0.06, "Comparing homogenized coarse flux vs. MPACT coarse flux");
}

TEST(homogenization, transportXS)
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;

    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group coarseGroup = file.getGroup("CMFD_CoarseMesh");
    HighFive::Group fineGroup = file.getGroup("CMFD_FineMesh");

    CMFDData<AssemblySpace> coarseData(coarseGroup);
    FineMeshData<AssemblySpace> fineData(fineGroup);

    FineMeshData<AssemblySpace>::View2D calculatedCoarseXS = fineData.homogenizeXS(fineData.transportXS);
    compare2DViews<PetscScalar, AssemblySpace>(calculatedCoarseXS, coarseData.transportXS, 2e-4, 1e-2, "Comparing homogenized coarse XS vs. MPACT coarse XS");
}

TEST(homogenization, removalXS)
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;

    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group coarseGroup = file.getGroup("CMFD_CoarseMesh");
    HighFive::Group fineGroup = file.getGroup("CMFD_FineMesh");

    CMFDData<AssemblySpace> coarseData(coarseGroup);
    FineMeshData<AssemblySpace> fineData(fineGroup);

    FineMeshData<AssemblySpace>::View2D calculatedCoarseXS = fineData.homogenizeXS(fineData.removalXS);
    compare2DViews<PetscScalar, AssemblySpace>(calculatedCoarseXS, coarseData.removalXS, 2e-3, 1e-2, "Comparing homogenized coarse removal XS vs. MPACT coarse removal XS");
}

TEST(homogenization, scatterXS)
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;

    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group coarseGroup = file.getGroup("CMFD_CoarseMesh");
    HighFive::Group fineGroup = file.getGroup("CMFD_FineMesh");

    CMFDData<AssemblySpace> coarseData(coarseGroup);
    FineMeshData<AssemblySpace> fineData(fineGroup);

    FineMeshData<AssemblySpace>::View3D calculatedCoarseXS = fineData.homogenizeScatteringXS();
    compare3DViews<PetscScalar, AssemblySpace>(calculatedCoarseXS, coarseData.scatteringXS, 0.002, 7e-3, "Comparing homogenized coarse scattering XS vs. MPACT coarse scattering XS");
}

TEST(homogenization, chi)
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;

    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group coarseGroup = file.getGroup("CMFD_CoarseMesh");
    HighFive::Group fineGroup = file.getGroup("CMFD_FineMesh");

    CMFDData<AssemblySpace> coarseData(coarseGroup);
    FineMeshData<AssemblySpace> fineData(fineGroup);

    FineMeshData<AssemblySpace>::View2D calculatedCoarseChi = fineData.homogenizeChi();
    compare2DViews<PetscScalar, AssemblySpace>(calculatedCoarseChi, coarseData.chi, 1e-14, 1e-14, "Comparing homogenized coarse chi vs. MPACT coarse chi");
}