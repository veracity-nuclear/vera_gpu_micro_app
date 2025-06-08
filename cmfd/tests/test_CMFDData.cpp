/*
  This file
*/
#include "PetscKokkosTestEnvironment.hpp"
#include "CMFDData.hpp"

TEST(readData, testIsKokkosDualView)
{
    using DualViewType = Kokkos::DualView<double*>;
    EXPECT_TRUE(isKokkosDualView<DualViewType>::value) << "isKokkosDualView should return true for Kokkos::DualView";

    using NonDualViewType = Kokkos::View<double*>;
    EXPECT_FALSE(isKokkosDualView<NonDualViewType>::value) << "isKokkosDualView should return false for Kokkos::View";
}

TEST(readData, testHDF5ToKokkosView)
{
    std::string filename = "data/pin_7g_16a_3p_serial.h5";
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::DataSet H5Dataset = file.getDataSet("CMFD_CoarseMesh/flux");
    std::vector<size_t> dims = H5Dataset.getDimensions();

    std::vector<std::vector<double>> originalData(dims[0], std::vector<double>(dims[1], 0.0));
    H5Dataset.read(originalData);

    auto dualView = HDF5ToKokkosView<Kokkos::DualView<double**>>(H5Dataset, "flux");

    // Check if the data is correctly copied
    auto h_view = dualView.view_host();
    for (size_t i = 0; i < h_view.extent(0); ++i) {
        for (size_t j = 0; j < h_view.extent(1); ++j) {
            EXPECT_DOUBLE_EQ(h_view(i, j), originalData[i][j]) << "Error copying data from HDF5 to DualView Host";
        }
    }

    // Check if the data is correctly copied to the device
    auto d_view = dualView.view_device();
    Kokkos::View<double**>::HostMirror check_d_view = Kokkos::create_mirror_view(d_view);
    Kokkos::deep_copy(check_d_view, d_view);
    for (size_t i = 0; i < check_d_view.extent(0); ++i) {
        for (size_t j = 0; j < check_d_view.extent(1); ++j) {
            EXPECT_DOUBLE_EQ(check_d_view(i, j), originalData[i][j]) << "Error copying data from HDF5 to DualView Device";
        }
    }
}