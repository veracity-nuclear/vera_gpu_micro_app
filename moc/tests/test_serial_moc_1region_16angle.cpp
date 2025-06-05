#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "eigen_solver.hpp"
#include "argument_parser.hpp"

TEST(BasicTest, cart_1region_7g_16a_3p_serial) {
    const char* raw_args[] = {"exe", "data/cart_1region_7g_16a_3p_serial.h5", "data/c5g7.xsl"};
    char** args = const_cast<char**>(raw_args);
    ArgumentParser parser("test_serial_moc_1region_16angle",
                          "Test Serial MOC with 1 region, 16 angles, and 7 energy groups");
    parser.add_argument("input_file", "Input HDF5 file");
    parser.add_argument("cross_sections", "Cross section library file");
    parser.parse(3, args);
    BaseMOC* sweeper = new SerialMOC(parser);
    EigenSolver solver(parser.get_args("test_serial_moc_1region_16angle.exe"), sweeper);
    solver.solve();
    EXPECT_NEAR(solver.keff(), 0.73822784, 1.0e-7);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
