#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "conduction_serial.hpp"
#include "materials.hpp"

/* The SMR test running with Kokkos::OpenMP runs 184,730 conduction solves and checks that the
   results are identical (bitwise) to the Kokkos::Serial execution of the same problem. */

const std::string filename = std::string(TEST_DATA_DIR) + "/smr.h5";

template<typename ExecutionSpace>
typename Conduction<ExecutionSpace>::ConductionResults run_smr(const std::vector<std::string>& state_groups) {
    std::vector<double> radii = {0.0, 0.4096, 0.4180, 0.4750}; // fuel, gap, clad radii in cm
    double height = 11.951; // cm

    std::vector<std::shared_ptr<Solid>> materials = {
        std::make_shared<UO2>(),
        std::make_shared<Zircaloy>()
    };

    Conduction<ExecutionSpace> conduction_solver(radii, height, materials);
    conduction_solver.load_pin_data(filename, state_groups);
    return conduction_solver.solve_all_pins();
}

TEST(SMR_OpenMP, ConductionSolveMatchesSerial) {
#ifndef KOKKOS_ENABLE_OPENMP
    GTEST_SKIP() << "Kokkos was built without OpenMP";
#else
    std::vector<std::string> state_groups = {
        "/STATE_0001", "/STATE_0002", "/STATE_0003", "/STATE_0004", "/STATE_0005",
        "/STATE_0006", "/STATE_0007", "/STATE_0008", "/STATE_0009", "/STATE_0010",
    };

    auto results = run_smr<Kokkos::OpenMP>(state_groups);
    auto serial = run_smr<Kokkos::Serial>(state_groups);

    EXPECT_EQ(results.total_pins, 18473 * state_groups.size());
    EXPECT_EQ(results.average_temperatures.size(), results.total_pins * 2); // 2 nodes per pin
    EXPECT_EQ(results.interface_temperatures.size(), results.total_pins);

    // Each pin is solved independently, so the parallel result must match serial exactly
    ASSERT_EQ(results.average_temperatures.size(), serial.average_temperatures.size());
    for (size_t i = 0; i < results.average_temperatures.size(); ++i) {
        ASSERT_EQ(results.average_temperatures[i], serial.average_temperatures[i]) << "average temperature " << i;
    }
    ASSERT_EQ(results.interface_temperatures.size(), serial.interface_temperatures.size());
    for (size_t i = 0; i < results.total_pins; ++i) {
        ASSERT_EQ(results.interface_temperatures[i].size(), 4u); // fuel CL, fuel surface, clad inner, clad outer
        ASSERT_EQ(results.interface_temperatures[i], serial.interface_temperatures[i]) << "pin " << i;
        for (double temp : results.interface_temperatures[i]) {
            EXPECT_GT(temp, 0.0);
        }
    }

    std::cout << "Solved " << results.total_pins << " pins in "
              << results.elapsed_time_ms << " ms using ConductionOpenMP with "
              << Kokkos::OpenMP().concurrency() << " threads (serial: "
              << serial.elapsed_time_ms << " ms)" << std::endl;
#endif
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
