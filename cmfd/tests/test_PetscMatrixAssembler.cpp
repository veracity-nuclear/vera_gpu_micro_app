/*
  This file tests if the PetscMatrixAssembler works as intended.
*/
#include "PetscKokkosTestEnvironment.hpp"
#include "PetscMatrixAssembler.hpp"

TEST(simpleMatrixAssembler, initialize)
{
    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    const HighFive::Group CMFDCoarseMesh = file.getGroup("CMFD_CoarseMesh");

    // Get data the standard way
    size_t firstCell, lastCell, nEnergyGroups;
    std::vector<PetscScalar> volume;
    std::vector<std::vector<PetscScalar>> chi, Dhat, Dtilde, nuFissionXs, pastFlux, removalXs;
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
    CMFDCoarseMesh.getDataSet("flux").read(pastFlux);
    CMFDCoarseMesh.getDataSet("removal XS").read(removalXs);
    CMFDCoarseMesh.getDataSet("scattering XS").read(scatteringXs);
    CMFDCoarseMesh.getDataSet("surf2cell").read(surf2Cell);

    size_t nCells = lastCell - firstCell + 1;
    size_t nSurfaces = surf2Cell.size();

    // Get data using the Assembler
    SimpleMatrixAssembler assembler(CMFDCoarseMesh);

    EXPECT_EQ(assembler.nCells, nCells) << "Number of cells mismatch";
    EXPECT_EQ(assembler.nSurfaces, nSurfaces) << "Number of surfaces mismatch";
    EXPECT_EQ(assembler.nGroups, nEnergyGroups) << "Number of energy groups mismatch";

    compare2DViewAndVector(assembler.chi.view_host(), chi, "Chi data mismatch");
    compare2DViewAndVector(assembler.dHat.view_host(), Dhat, "Dhat data mismatch");
    compare2DViewAndVector(assembler.dTilde.view_host(), Dtilde, "Dtilde data mismatch");
    compare2DViewAndVector(assembler.nuFissionXS.view_host(), nuFissionXs, "Nu Fission XS data mismatch");
    compare2DViewAndVector(assembler.pastFlux.view_host(), pastFlux, "Past Flux data mismatch");
    compare2DViewAndVector(assembler.removalXS.view_host(), removalXs, "Removal XS data mismatch");

    auto h_volume = assembler.volume.view_host();
    for (size_t i = 0; i < nCells; ++i) {
        EXPECT_DOUBLE_EQ(h_volume(i), volume[i]) << "Volume data mismatch at index " << i;
    }

    auto h_surf2Cell = assembler.surf2Cell.view_host();
    for (size_t i = 0; i < nSurfaces; ++i) {
        EXPECT_EQ(h_surf2Cell(i, 0), surf2Cell[i][0]) << "Surf2Cell data mismatch at index " << i << ", first element";
        EXPECT_EQ(h_surf2Cell(i, 1), surf2Cell[i][1]) << "Surf2Cell data mismatch at index " << i << ", second element";
    }

    auto h_scatteringXs = assembler.scatteringXS.view_host();
    for (size_t eg1 = 0; eg1 < nEnergyGroups; ++eg1) {
        for (size_t eg2 = 0; eg2 < nEnergyGroups; ++eg2) {
            for (size_t i = 0; i < nCells; ++i) {
                EXPECT_DOUBLE_EQ(h_scatteringXs(eg1, eg2, i), scatteringXs[eg1][eg2][i])
                    << "Scattering XS data mismatch at (" << eg1 << ", " << eg2 << ", " << i << ")";
            }
        }
    }
}