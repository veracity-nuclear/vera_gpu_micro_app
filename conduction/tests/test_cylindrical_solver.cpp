#include <gtest/gtest.h>
#include "cylinder_node.hpp"
#include "cylindrical_solver.hpp"

TEST(CylindricalSolverTest, ConstructorValid) {
    double height = 1.0;
    CylinderNode node1(height, 0.01, 0.02);
    CylinderNode node2(height, 0.02, 0.05);
    std::vector<CylinderNode> nodes = {node1, node2};
    EXPECT_NO_THROW(CylindricalSolver solver(nodes)); // Valid cylindrical nodes
}

TEST(CylindricalSolverTest, ConstructorInvalid) {
    double height = 1.0;
    CylinderNode node1(height, 0.02, 0.05);
    CylinderNode node2(height, 0.01, 0.02);
    std::vector<CylinderNode> nodes = {node1, node2};
    EXPECT_THROW(CylindricalSolver solver(nodes), std::runtime_error); // Invalid cylindrical nodes
}

TEST(CylindricalSolverTest, TemperatureDistribution_1Region) {
    double height = 0.025; // m
    std::vector<double> radii = {0.0, 0.008}; // fuel radius in m
    std::vector<CylinderNode> nodes = {
        CylinderNode(height, radii[0], radii[1])
    };
    std::vector<std::shared_ptr<SolidMaterial>> materials = {
        std::make_shared<UO2>()
    };

    CylindricalSolver solver(nodes);
    std::vector<double> qdot = {3.8e6};  // W/m^3
    double T_outer = 600.0; // K

    std::vector<double> avg_temps = solver.solve_temperatures(qdot, materials, T_outer);
    std::vector<double> interface_temps = solver.get_interface_temperatures();
    double T_fuel_cl = interface_temps[0];

    // analytical solution for temperature at fuel centerline
    // T(r=0) = T_outer + qdot / (4 * k) * (r_out^2 - r_in^2)
    double T_fuel_cl_analytical = 611.855227; // K

    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-6);

    EXPECT_EQ(avg_temps.size(), 1);
    EXPECT_NEAR(avg_temps[0], 608.891420, 1e-6);
}

TEST(CylindricalSolverTest, TemperatureDistribution_FuelPin_3Regions) {
    double height = 0.11951; // m
    std::vector<double> radii = {0.0, 0.004096, 0.004180, 0.004750}; // fuel, gap, clad radii in m
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

    double qdot_fuel = 1000 / nodes[0].get_volume(); // W/m^3

    CylindricalSolver solver(nodes);
    std::vector<double> qdot = {qdot_fuel, 0.0, 0.0};  // W/m^3
    double T_outer = 600.0; // K

    std::vector<double> avg_temps = solver.solve_temperatures(qdot, materials, T_outer);
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
    double T_fuel_cl_analytical = 882.856055; // K
    double T_fuel_outer_analytical = 715.018591; // K
    double T_clad_inner_analytical = 610.289235; // K
    double T_clad_outer_analytical = 600.000000; // K

    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-6);
    EXPECT_NEAR(T_fuel_outer, T_fuel_outer_analytical, 1e-6);
    EXPECT_NEAR(T_clad_inner, T_clad_inner_analytical, 1e-6);
    EXPECT_NEAR(T_clad_outer, T_clad_outer_analytical, 1e-6);

    EXPECT_EQ(avg_temps.size(), 3);
    EXPECT_NEAR(avg_temps[0], 840.896689, 1e-6);
    EXPECT_NEAR(avg_temps[1], 662.388162, 1e-6);
    EXPECT_NEAR(avg_temps[2], 604.980316, 1e-6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
