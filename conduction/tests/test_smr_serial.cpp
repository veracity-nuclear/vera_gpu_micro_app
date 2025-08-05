#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "cylinder_node.hpp"
#include "cylindrical_solver_serial.hpp"
#include "hdf5_utils.hpp"

/* The SMR test running in serial runs 184,730 conduction solves in 412 ms. */

const std::string filename = std::string(TEST_DATA_DIR) + "/smr.h5";

TEST(SMR_Serial, ConductionSolve) {
    std::vector<double> radii = {0.0, 0.4096, 0.4180, 0.4750}; // fuel, gap, clad radii in cm
    double height = 11.951; // cm

    std::vector<std::string> state_groups = {
        "/STATE_0001", "/STATE_0002", "/STATE_0003", "/STATE_0004", "/STATE_0005",
        "/STATE_0006", "/STATE_0007", "/STATE_0008", "/STATE_0009", "/STATE_0010",
    };

    std::vector<double> all_pin_powers;
    std::vector<double> all_clad_surf_temps;

    for (const auto& group : state_groups) {
        double total_power = read_hdf5_scalar(filename, group + "/total_power");
        double power_percent = read_hdf5_scalar(filename, group + "/power") * 0.01; // convert to fraction

        FlatHDF5Data pin_powers = read_flat_hdf5_dataset(filename, group + "/pin_powers");
        double scaling = total_power * power_percent / std::accumulate(pin_powers.data.begin(), pin_powers.data.end(), 0.0);
        for (auto& p : pin_powers.data) p *= scaling;
        EXPECT_NEAR(std::accumulate(pin_powers.data.begin(), pin_powers.data.end(), 0.0), total_power * power_percent, 1e-3);

        FlatHDF5Data clad_surf_temps = read_flat_hdf5_dataset(filename, group + "/pin_max_clad_surface_temp");
        for (auto& T : clad_surf_temps.data) T += 273.15;  // convert to K

        // Append to global vectors
        all_pin_powers.insert(all_pin_powers.end(), pin_powers.data.begin(), pin_powers.data.end());
        all_clad_surf_temps.insert(all_clad_surf_temps.end(), clad_surf_temps.data.begin(), clad_surf_temps.data.end());
    }

    assert(all_pin_powers.size() == all_clad_surf_temps.size());
    size_t N = all_pin_powers.size();
    EXPECT_EQ(N, 18473 * state_groups.size());


    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        std::vector<std::shared_ptr<CylinderNode>> nodes = {
            std::make_shared<CylinderNode>(height, radii[0], radii[1]),
            std::make_shared<CylinderNode>(height, radii[2], radii[3])
        };
        std::vector<std::shared_ptr<Solid>> materials = {
            std::make_shared<UO2>(),
            std::make_shared<Zircaloy>()
        };
        CylindricalSolverSerial solver(nodes, materials);
        double T_outer = all_clad_surf_temps[i];

        // Create Kokkos::View for qdot
        CylindricalSolverSerial::DoubleView qdot("qdot", nodes.size());
        auto h_qdot = Kokkos::create_mirror_view(qdot);
        for (size_t j = 0; j < nodes.size(); ++j) {
            h_qdot(j) = 0.0;
        }
        h_qdot(0) = all_pin_powers[i] / nodes[0]->get_volume(); // only node with fuel has heat gen
        Kokkos::deep_copy(qdot, h_qdot);

        CylindricalSolverSerial::DoubleView Tavg = solver.solve(qdot, T_outer);
        CylindricalSolverSerial::DoubleView interface_temps = solver.get_interface_temperatures();

        // Copy to host for verification
        auto h_Tavg = Kokkos::create_mirror_view(Tavg);
        Kokkos::deep_copy(h_Tavg, Tavg);

        double sum_temps = 0.0;
        for (size_t j = 0; j < h_Tavg.extent(0); ++j) {
            sum_temps += h_Tavg(j);
        }
        EXPECT_GT(sum_temps, 0.0); // use result to prevent compiler optimization
        EXPECT_EQ(h_Tavg.extent(0), nodes.size());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: "
          << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
          << " ms" << std::endl;
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
