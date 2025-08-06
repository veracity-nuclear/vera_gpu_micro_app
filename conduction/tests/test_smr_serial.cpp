#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "conduction_serial.hpp"
#include "materials.hpp"

/* The SMR test running in serial runs 184,730 conduction solves using a parallel_for kernel. */

const std::string filename = std::string(TEST_DATA_DIR) + "/smr.h5";

TEST(SMR_Serial, ConductionSolve) {
    std::vector<double> radii = {0.0, 0.4096, 0.4180, 0.4750}; // fuel, gap, clad radii in cm
    double height = 11.951; // cm

    std::vector<std::shared_ptr<Solid>> materials = {
        std::make_shared<UO2>(),
        std::make_shared<Zircaloy>()
    };

    std::vector<std::string> state_groups = {
        "/STATE_0001", "/STATE_0002", "/STATE_0003", "/STATE_0004", "/STATE_0005",
        "/STATE_0006", "/STATE_0007", "/STATE_0008", "/STATE_0009", "/STATE_0010",
    };

    // Create conduction solver
    ConductionSerial conduction_solver(radii, height, materials);

    // Load pin data from HDF5 file
    conduction_solver.load_pin_data(filename, state_groups);

    // Solve all pins
    ConductionSerial::ConductionResults results = conduction_solver.solve_all_pins();

    // Verify results
    EXPECT_EQ(results.total_pins, 18473 * state_groups.size());
    EXPECT_GT(results.elapsed_time_ms, 0.0);

    // Verify that we got results for all pins
    EXPECT_EQ(results.average_temperatures.size(), results.total_pins * 2); // 2 nodes per pin
    EXPECT_EQ(results.interface_temperatures.size(), results.total_pins);

    // Verify that all temperature sums are positive (prevents compiler optimization)
    for (size_t i = 0; i < results.total_pins; ++i) {
        EXPECT_GT(results.interface_temperatures[i], 0.0);
    }

    std::cout << "Solved " << results.total_pins << " pins in "
              << results.elapsed_time_ms << " ms using restructured ConductionSerial" << std::endl;
}

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {
        ::testing::InitGoogleTest(&argc, argv);
        auto result = RUN_ALL_TESTS();
        Kokkos::finalize();
        return result;
    }
}
