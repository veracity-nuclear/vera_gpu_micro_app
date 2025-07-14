#include <gtest/gtest.h>
#include "cylinder_node.hpp"
#include "cylindrical_solver.hpp"
#include "hdf5_utils.hpp"

/* The SMR test running in serial runs 184,730 conduction solves in 0.16 seconds. */

const std::string filename = std::string(TEST_DATA_DIR) + "/smr.h5";

TEST(SMR_Serial, ConductionSolve) {
    std::vector<double> radii = {0.0, 0.004096, 0.004180, 0.004750}; // fuel, gap, clad radii in m
    double height = 0.11951; // m
    std::vector<CylinderNode> nodes = {
        CylinderNode(height, radii[0], radii[1]),
        CylinderNode(height, radii[1], radii[2]),
        CylinderNode(height, radii[2], radii[3])
    };
    std::vector<std::shared_ptr<SolidMaterial>> materials = {
        std::make_shared<UO2>(),
        std::make_shared<Helium>(),
        std::make_shared<Zircaloy>()
    };
    CylindricalSolver solver(nodes);

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

    std::vector<std::vector<double>> qdot(N, std::vector<double>(nodes.size(), 0.0));
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        double T_outer = all_clad_surf_temps[i];
        qdot[i][0] = all_pin_powers[i] / nodes[0].get_volume(); // only node with fuel has heat gen
        std::vector<double> Tavg = solver.solve_temperatures(qdot[i], materials, T_outer);
        EXPECT_GT(std::accumulate(Tavg.begin(), Tavg.end(), 0.0), 0.0); // use result to prevent compiler optimization
        EXPECT_EQ(Tavg.size(), nodes.size());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: "
          << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
          << " ms" << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
