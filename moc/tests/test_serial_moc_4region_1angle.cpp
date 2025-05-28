#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "serial_moc.hpp"

TEST(BasicTest, cart_4region_7g_1a_1p_serial) {
    const std::vector<std::string> args = {"exe", "data/cart_4region_7g_1a_1p_serial.h5", "data/c5g7.xsl"};
    double result = serial_moc_sweep(args);
    EXPECT_NEAR(result, 0.73822736, 1.0e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
