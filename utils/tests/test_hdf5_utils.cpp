#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include "hdf5_utils.hpp"

TEST(HDF5UtilsTest, SingleDimensionalDatasetRead) {
    const std::string filename = std::string(TEST_DATA_DIR) + "/smr.h5";
    const std::string group_name = "/STATE_0001";
    const std::string dataset_name = "pin_powers";

    try {

    } catch (const HighFive::Exception& err) {
        std::cerr << "[ERROR] " << err.what() << "\n";
        FAIL();
    }
}

TEST(HDF5UtilsTest, MultiDimensionalDatasetRead) {
    const std::string filename = std::string(TEST_DATA_DIR) + "/smr.h5";
    const std::string group_name = "/STATE_0001";
    const std::string dataset_name = "pin_powers";

    FlatHDF5Data pin_powers;
    try {
        pin_powers = read_flat_hdf5_dataset(filename, group_name + "/pin_powers");
    } catch (const HighFive::Exception& err) {
        std::cerr << "[ERROR] " << err.what() << "\n";
        FAIL();
    }

    ASSERT_EQ(pin_powers.shape.size(), 4) << "Expected a 4-D dataset";
    ASSERT_EQ(pin_powers.shape[0], 7) << "Expected first dimension to be 7";
    ASSERT_EQ(pin_powers.shape[1], 7) << "Expected second dimension to be 7";
    ASSERT_EQ(pin_powers.shape[2], 29) << "Expected third dimension to be 29";
    ASSERT_EQ(pin_powers.shape[3], 13) << "Expected fourth dimension to be 13";
    ASSERT_EQ(pin_powers.data.size(), pin_powers.size()) << "Data size mismatch with total size";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
