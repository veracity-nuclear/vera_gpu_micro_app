#include <gtest/gtest.h>
#include "cylinder_node.hpp"
#include "cylindrical_solver_serial.hpp"

TEST(CylindricalSolverTest, ConstructorValid) {
    double height = 1.0;
    CylinderNode node1(height, 0.01, 0.02);
    CylinderNode node2(height, 0.02, 0.05);
    std::vector<std::shared_ptr<CylinderNode>> nodes = {
        std::make_shared<CylinderNode>(node1),
        std::make_shared<CylinderNode>(node2)
    };
    std::vector<std::shared_ptr<Solid>> materials = {
        std::make_shared<UO2>(),
        std::make_shared<Zircaloy>()
    };
    EXPECT_NO_THROW(CylindricalSolverSerial solver(nodes, materials)); // Valid cylindrical nodes
}

TEST(CylindricalSolverTest, ConstructorInvalid) {
    double height = 1.0;
    CylinderNode node1(height, 0.02, 0.05);
    CylinderNode node2(height, 0.01, 0.02);
    std::vector<std::shared_ptr<CylinderNode>> nodes = {
        std::make_shared<CylinderNode>(node1),
        std::make_shared<CylinderNode>(node2)
    };
    std::vector<std::shared_ptr<Solid>> materials = {
        std::make_shared<UO2>(),
        std::make_shared<Zircaloy>()
    };
    EXPECT_THROW(CylindricalSolverSerial solver(nodes, materials), std::runtime_error); // Invalid cylindrical nodes
}

TEST(CylindricalSolverTest, TemperatureDistribution_1Region) {
    double height = 2.5; // cm
    std::vector<double> radii = {0.0, 0.8}; // fuel radius in cm
    std::vector<std::shared_ptr<CylinderNode>> nodes = {
        std::make_shared<CylinderNode>(height, radii[0], radii[1])
    };
    std::vector<std::shared_ptr<Solid>> materials = {
        std::make_shared<UO2>()
    };

    CylindricalSolverSerial solver(nodes, materials);

    std::vector<double> qdot(1);
    qdot[0] = 100 / nodes[0]->get_volume();  // W/cm^3

    double T_outer = 600.0; // K

    std::vector<double> avg_temps = solver.solve(qdot, T_outer);
    std::vector<double> interface_temps = solver.get_interface_temperatures();

    double T_fuel_cl = interface_temps[0];

    // analytical solution for temperature at fuel centerline
    // T(r=0) = T_outer + qdot / (4 * k) * (r_out^2 - r_in^2)
    double T_fuel_cl_analytical = 665.267006; // K

    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-6);

    EXPECT_EQ(avg_temps.size(), 1);
    EXPECT_NEAR(avg_temps[0], 648.950255, 1e-6);
}

TEST(CylindricalSolverTest, TemperatureDistribution_FuelPin_10Regions) {
    double height = 11.951; // cm

    std::vector<double> fuel_radii = {0.0000, 0.4096}; // fuel radii in cm
    std::vector<double> clad_radii = {0.4180, 0.4750}; // clad radii in cm

    double N_fuel_regions = 8; // Number of fuel regions
    double N_clad_regions = 2; // Number of clad regions
    double pin_total_power = 1000.0; // W
    double r0, r1, dr;

    std::vector<std::shared_ptr<CylinderNode>> nodes = {};
    std::vector<std::shared_ptr<Solid>> materials = {};
    std::vector<double> power = {}; // W

    r0 = fuel_radii[0];
    r1 = fuel_radii[1];
    dr = (r1 - r0) / N_fuel_regions;
    for (size_t i = 0; i < N_fuel_regions; ++i) {
        double inner_radius = r0 + i * dr;
        double outer_radius = r0 + (i + 1) * dr;
        nodes.push_back(std::make_shared<CylinderNode>(height, inner_radius, outer_radius));
        materials.push_back(std::make_shared<UO2>());
        power.push_back(1.0 - std::pow(inner_radius / r1, 2)); // W
    }

    // normalize power to 1000 W
    double normalization_factor = std::accumulate(power.begin(), power.end(), 0.0);
    for (auto &p : power) {
        p = (p / normalization_factor) * pin_total_power; // W
    }

    r0 = clad_radii[0];
    r1 = clad_radii[1];
    dr = (r1 - r0) / N_clad_regions;
    for (size_t i = 0; i < N_clad_regions; ++i) {
        double inner_radius = r0 + i * dr;
        double outer_radius = r0 + (i + 1) * dr;
        nodes.push_back(std::make_shared<CylinderNode>(height, inner_radius, outer_radius));
        materials.push_back(std::make_shared<Zircaloy>());
        power.push_back(0.0); // W
    }

    std::vector<double> qdot_vec(nodes.size(), 0.0); // W/m^3
    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        qdot_vec[i] = power[i] / nodes[i]->get_volume(); // W/m^3
    }

    CylindricalSolverSerial solver(nodes, materials);
    double T_outer = 600.0; // K

    std::vector<double> avg_temps = solver.solve(qdot_vec, T_outer);
    std::vector<double> interface_temps = solver.get_interface_temperatures();

    double T_fuel_cl = interface_temps[0];
    double T_fuel_outer = interface_temps[8];
    double T_clad_inner = interface_temps[9];
    double T_clad_outer = interface_temps[11];

    // analytical solution for temperature at fuel centerline
    // T(r=0.) = T_outer + q * (R_fuel + R_gap + R_clad)
    // T(r=r1) = T_outer + q * (R_gap + R_clad)
    // T(r=r2) = T_outer + q * R_clad
    // T(r=r3) = T_outer
    double T_fuel_cl_analytical    = 982.802240; // K
    double T_fuel_outer_analytical = 611.507420; // K
    double T_clad_inner_analytical = 610.334927; // K
    double T_clad_outer_analytical = 600.000000; // K

    // Test interface temperatures
    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-6);
    EXPECT_NEAR(T_fuel_outer, T_fuel_outer_analytical, 1e-6);
    EXPECT_NEAR(T_clad_inner, T_clad_inner_analytical, 1e-6);
    EXPECT_NEAR(T_clad_outer, T_clad_outer_analytical, 1e-6);

    // Test average temperatures
    EXPECT_EQ(avg_temps.size(), 10);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
