#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base_moc.hpp"
#include "kokkos_moc.hpp"
#include "eigen_solver.hpp"
#include "argument_parser.hpp"

TEST(BasicTest, 7x7_7g_16a_3p_kokkos_single) {
    const char* raw_args[] = {"exe", "data/7x7_7g_16a_3p_serial.h5", "data/c5g7.xsl", "--sweeper", "kokkos", "--device", "openmp", "--kokkos-num-threads=4", "--precision", "single"};
    char** args = const_cast<char**>(raw_args);
    int argc = 10;
    Kokkos::initialize(argc, args);
    {
        auto parser = ArgumentParser::vera_gpu_moc_parser(raw_args[0]);
        parser.parse(argc, args);
        std::shared_ptr<BaseMOC> sweeper(new KokkosMOC<Kokkos::OpenMP, float>(parser));
        EigenSolver solver(parser, sweeper);
        solver.solve();
        EXPECT_NEAR(solver.keff(), 1.340878337, 1.0e-7);
    }
    Kokkos::finalize();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
