#include <gtest/gtest.h>
#include "cylinder_node.hpp"
#include "cylindrical_solver.hpp"

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
    EXPECT_NO_THROW(CylindricalSolver solver(nodes, materials)); // Valid cylindrical nodes
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
    EXPECT_THROW(CylindricalSolver solver(nodes, materials), std::runtime_error); // Invalid cylindrical nodes
}

TEST(CylindricalSolverTest, TemperatureDistribution_1Region) {
    double height = 0.025; // m
    std::vector<double> radii = {0.0, 0.008}; // fuel radius in m
    std::vector<std::shared_ptr<CylinderNode>> nodes = {
        std::make_shared<CylinderNode>(height, radii[0], radii[1])
    };
    std::vector<std::shared_ptr<Solid>> materials = {
        std::make_shared<UO2>()
    };

    CylindricalSolver solver(nodes, materials);
    std::vector<double> qdot = {3.8e6};  // W/m^3
    double T_outer = 600.0; // K

    std::vector<double> avg_temps = solver.solve(qdot, T_outer);
    std::vector<double> interface_temps = solver.get_interface_temperatures();
    double T_fuel_cl = interface_temps[0];

    // analytical solution for temperature at fuel centerline
    // T(r=0) = T_outer + qdot / (4 * k) * (r_out^2 - r_in^2)
    double T_fuel_cl_analytical = 611.857359; // K

    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-6);

    EXPECT_EQ(avg_temps.size(), 1);
    EXPECT_NEAR(avg_temps[0], 608.893019, 1e-6);
}

TEST(CylindricalSolverTest, TemperatureDistribution_FuelPin_3Regions) {
    double height = 0.11951; // m
    std::vector<double> radii = {0.0, 0.004096, 0.004180, 0.004750}; // radii in m
    std::vector<std::shared_ptr<CylinderNode>> nodes = {
        std::make_shared<CylinderNode>(height, radii[0], radii[1]),
        std::make_shared<CylinderNode>(height, radii[2], radii[3])
    };
    std::vector<std::shared_ptr<Solid>> materials = {
        std::make_shared<UO2>(),
        std::make_shared<Zircaloy>()
    };

    double qdot_fuel = 1000 / nodes[0]->get_volume(); // W/m^3

    CylindricalSolver solver(nodes, materials);
    std::vector<double> qdot = {qdot_fuel, 0.0};  // W/m^3
    double T_outer = 600.0; // K

    std::vector<double> avg_temps = solver.solve(qdot, T_outer);
    std::vector<double> interface_temps = solver.get_interface_temperatures();
    double T_fuel_cl = interface_temps[0];
    double T_fuel_outer = interface_temps[1];
    double T_clad_inner = interface_temps[2];
    double T_clad_outer = interface_temps[3];

    // analytical solution for temperature at fuel centerline
    // T(r=0.) = T_outer + q * (R_fuel + R_gap + R_clad)
    // T(r=r1) = T_outer + q * (R_gap + R_clad)
    // T(r=r2) = T_outer + q * R_clad
    // T(r=r3) = T_outer
    double T_fuel_cl_analytical = 888.662; // K
    double T_fuel_outer_analytical = 719.112; // K
    double T_clad_inner_analytical = 610.340; // K
    double T_clad_outer_analytical = 600.000; // K

    // Test interface temperatures
    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-3);
    EXPECT_NEAR(T_fuel_outer, T_fuel_outer_analytical, 1e-3);
    EXPECT_NEAR(T_clad_inner, T_clad_inner_analytical, 1e-3);
    EXPECT_NEAR(T_clad_outer, T_clad_outer_analytical, 1e-3);

    // Test average temperatures
    EXPECT_EQ(avg_temps.size(), 2);
    EXPECT_NEAR(avg_temps[0], 846.274, 1e-3);
    EXPECT_NEAR(avg_temps[1], 605.005, 1e-3);

    // Test thermal expansion effects
    EXPECT_NEAR(nodes[0]->get_inner_radius(), 0.00000000, 1e-6);
    EXPECT_NEAR(nodes[0]->get_outer_radius(), 0.00410608, 1e-6);
    EXPECT_NEAR(nodes[1]->get_inner_radius(), 0.00418834, 1e-6);
    EXPECT_NEAR(nodes[1]->get_outer_radius(), 0.00475947, 1e-6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
