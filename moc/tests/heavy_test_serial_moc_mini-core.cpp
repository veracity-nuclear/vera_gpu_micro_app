#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "eigen_solver.hpp"
#include "argument_parser.hpp"

TEST(BasicTest, mini_core_7g_16a_3p_serial) {
    const char* raw_args[] = {"exe", "data/mini-core_7g_16a_3p_serial.h5", "data/c5g7.xsl"};
    char** args = const_cast<char**>(raw_args);
    ArgumentParser parser("heavy_test_serial_moc_mini-core",
                          "Test Serial MOC with mini-core, 16 angles, and 7 energy groups");
    parser.add_argument("input_file", "Input HDF5 file");
    parser.add_argument("cross_sections", "Cross section library file");
    parser.parse(3, args);
    BaseMOC* sweeper = new SerialMOC(parser);
    EigenSolver solver(parser.get_args("heavy_test_serial_moc_mini-core.exe"), sweeper);
    solver.solve();
    EXPECT_NEAR(solver.keff(), 0.96030553, 1.0e-7);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
