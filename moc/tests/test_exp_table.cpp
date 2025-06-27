#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "exp_table.hpp"
#include "base_moc.hpp"
#include "kokkos_moc.hpp"
#include "eigen_solver.hpp"
#include "argument_parser.hpp"

TEST(BasicTest, test_exp_table) {
    ExpTable table;
    EXPECT_NEAR(table._min_val, -40.0, 1e-6);
    EXPECT_NEAR(table._max_val, 0.0, 1e-6);
    EXPECT_EQ(table._rdx, 1000);
    EXPECT_EQ(table._n_intervals, 40000);
    EXPECT_EQ(table._exp_table.size(), table._n_intervals + 1);
    for (int i = 0; i < table._n_intervals + 1; i++) {
        EXPECT_EQ(table._exp_table[i].size(), 2);
        double x = -40.0 + i * 0.001;
        EXPECT_NEAR(table.expt(x), 1.0 - std::exp(x), 1e-5);
    }
}

TEST(BasicTest, test_kokkos_exp_table) {
    const char* raw_args[] = {"exe", "data/pin_7g_16a_3p_serial.h5", "data/c5g7.xsl", "--sweeper", "kokkos", "--device", "serial"};
    int argc = 7;
    char** args = const_cast<char**>(raw_args);
    Kokkos::initialize(argc, args);
    {
        auto parser = ArgumentParser::vera_gpu_moc_parser(raw_args[0]);
        parser.parse(argc, args);
        std::shared_ptr<KokkosMOC<Kokkos::Serial>> sweeper(new KokkosMOC<Kokkos::Serial>(parser));

        for (int i = 0; i < 40001; i++) {
            double x = -40.0 + i * 0.001;
            EXPECT_NEAR(sweeper->_h_exp_table(i, 0) * x + sweeper->_h_exp_table(i, 1), 1.0 - std::exp(x), 1e-5)
                << "Failed for i = " << i << ", x = " << x;
        }
    }
    Kokkos::finalize();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
