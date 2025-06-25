#include <gtest/gtest.h>
#include "cylinder_node.hpp"
#include "cylindrical_solver.hpp"
#include "hdf5_utils.hpp"

const std::string filename = std::string(TEST_DATA_DIR) + "/smr.h5";
const std::string group_name = "/STATE_0001";

TEST(SMR_Serial, ConductionSolve) {
    std::vector<double> radii = {0.0, 0.004096, 0.004180, 0.004750}; // fuel, gap, clad radii in m
    double height = 0.11951; // m
    std::vector<CylinderNode> nodes = {
        CylinderNode(height, radii[0], radii[1]),
        CylinderNode(height, radii[1], radii[2]),
        CylinderNode(height, radii[2], radii[3])
    };
    CylindricalSolver solver(nodes);

    std::vector<double> k = {4.5, 0.2, 8.8};  // W/m-K
    double total_power = read_hdf5_scalar(filename, group_name + "/total_power");
    double power_percent = read_hdf5_scalar(filename, group_name + "/power") * 0.01; // Convert to fraction
    std::cout << "Total power: " << total_power << " W, Power percent: " << power_percent * 100 << "%" << std::endl;

    // Import normalized pin powers from HDF5 file
    FlatHDF5Data pin_powers = read_flat_hdf5_dataset(filename, group_name + "/pin_powers");
    pin_powers = pin_powers * (total_power * power_percent / std::accumulate(pin_powers.data.begin(), pin_powers.data.end(), 0.0)); // W
    EXPECT_NEAR(std::accumulate(pin_powers.data.begin(), pin_powers.data.end(), 0.0), 6.0606e6, 1e-3); // 6.06 MW

    // Import clad outer surface temperatures from HDF5 file
    FlatHDF5Data clad_surf_temps = read_flat_hdf5_dataset(filename, group_name + "/pin_max_clad_surface_temp") + 273.15; // Convert to Kelvin

    assert(pin_powers.size() == clad_surf_temps.size());
    size_t N = pin_powers.size();
    EXPECT_EQ(N, 18473); // Expected number of pins

    std::vector<std::vector<double>> qdot(N, std::vector<double>(nodes.size(), 0.0));
    for (size_t i = 0; i < N; ++i) {
        double T_outer = clad_surf_temps[i];
        double qdot_fuel = pin_powers[i] / nodes[0].get_volume(); // W/m^3
        qdot[i][0] = qdot_fuel;
        solver.solve_temperatures(qdot[i], k, T_outer);
        std::vector<double> Tavg = solver.get_average_temperatures();
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
