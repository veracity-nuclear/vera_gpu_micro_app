#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "eigen_solver.hpp"
#include "argument_parser.hpp"

TEST(BasicTest, cart_1region_7g_1a_1p_serial) {
    const char* raw_args[] = {"exe", "data/cart_1region_7g_1a_1p_serial.h5", "data/c5g7.xsl"};
    char** args = const_cast<char**>(raw_args);
    auto parser = ArgumentParser::vera_gpu_moc_parser(raw_args[0]);
    parser.parse(3, args);
    std::cout << parser.get_positional(0) << std::endl;
    BaseMOC* sweeper = new SerialMOC(parser);
    EigenSolver solver(parser.get_args("test_serial_moc_1region_1angle.exe"), sweeper);
    solver.solve();
    EXPECT_NEAR(solver.keff(), 0.73822767, 1.0e-7);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
