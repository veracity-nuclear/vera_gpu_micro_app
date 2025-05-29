#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "eigen_solver.hpp"

TEST(BasicTest, pin_7g_16a_3p_serial) {
    const std::vector<std::string> args = {"serial_moc.exe", "data/pin_7g_16a_3p_serial.h5", "data/c5g7.xsl"};
    double result = run_eigenvalue_iteration(args);
    EXPECT_NEAR(result, 1.32569524, 1.0e-7);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
