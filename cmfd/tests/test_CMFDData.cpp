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

TEST(readData, initialize)
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

    // Get data using the h_data
    const CMFDData<Kokkos::HostSpace> h_data(CMFDCoarseMesh);

    ASSERT_EQ(h_data.nCells, nCells) << "Number of cells mismatch";
    ASSERT_EQ(h_data.nSurfaces, nSurfaces) << "Number of surfaces mismatch";
    ASSERT_EQ(h_data.nGroups, nEnergyGroups) << "Number of energy groups mismatch";

    compare2DViewAndVector(h_data.chi, chi, "Chi data mismatch");
    compare2DViewAndVector(h_data.dHat, Dhat, "Dhat data mismatch");
    compare2DViewAndVector(h_data.dTilde, Dtilde, "Dtilde data mismatch");
    compare2DViewAndVector(h_data.nuFissionXS, nuFissionXs, "Nu Fission XS data mismatch");
    compare2DViewAndVector(h_data.pastFlux, pastFlux, "Past Flux data mismatch");
    compare2DViewAndVector(h_data.removalXS, removalXs, "Removal XS data mismatch");

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
    compare2DHostAndDevice(h_data.pastFlux, d_data.pastFlux, "Past Flux data mismatch on device");
    compare2DHostAndDevice(h_data.removalXS, d_data.removalXS, "Removal XS data mismatch on device");

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