#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "eigen_solver.hpp"
#include "argument_parser.hpp"

TEST(BasicTest, 7x7_7g_16a_3p_serial_double) {
    const char* raw_args[] = {"exe", "data/7x7_7g_16a_3p_serial.h5", "data/c5g7.xsl", "--precision", "double"};
    char** args = const_cast<char**>(raw_args);
    auto parser = ArgumentParser::vera_gpu_moc_parser(raw_args[0]);
    parser.parse(5, args);
    std::shared_ptr<BaseMOC> sweeper(new SerialMOC<double>(parser));
    EigenSolver solver(parser, sweeper);
    solver.solve();
    EXPECT_NEAR(solver.keff(), 1.34087940, 1.0e-7);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
