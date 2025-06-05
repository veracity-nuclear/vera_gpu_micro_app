#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "eigen_solver.hpp"

TEST(BasicTest, cart_1region_7g_1a_1p_serial) {
    BaseMOC* sweeper = new SerialMOC("data/cart_1region_7g_1a_1p_serial.h5", "data/c5g7.xsl");
    const std::vector<std::string> args = {};
    EigenSolver solver(args, sweeper);
    solver.solve();
    EXPECT_NEAR(solver.keff(), 0.73822767, 1.0e-7);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
