#include <highfive/H5File.hpp>
#include <gtest/gtest.h>

TEST(HighFiveTest, WriteReadScalar) {
    const std::string filename = "test_scalar.h5";
    const std::string dataset_name = "my_scalar";

    // Write a scalar value
    {
        HighFive::File file(filename, HighFive::File::Overwrite);
        int value_out = 42;
        file.createDataSet(dataset_name, value_out);
    }

    // Read the scalar value back
    {
        HighFive::File file(filename, HighFive::File::ReadOnly);
        int value_in;
        file.getDataSet(dataset_name).read(value_in);
        EXPECT_EQ(value_in, 42);
    }
}
