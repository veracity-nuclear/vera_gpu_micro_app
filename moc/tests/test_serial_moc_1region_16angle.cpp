#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "serial_moc.hpp"

TEST(BasicTest, cart_1region_7g_16a_3p_serial) {
    const std::vector<std::string> args = {"moc_tests", "data/cart_1region_7g_16a_3p_serial.h5", "data/c5g7.xsl"};
    double result = serial_moc_sweep(args);
    EXPECT_NEAR(result, 0.73822810, 1.0e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
