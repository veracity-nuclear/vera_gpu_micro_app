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

    CylindricalSolver solver(nodes);
    std::vector<double> k = {4.5};  // W/m-K
    std::vector<double> qdot = {3.8e6};  // W/m^3
    double T_outer = 600.0; // K

    std::vector<double> temps = solver.solve_temperatures(qdot, k, T_outer);
    double T_fuel_cl = temps[0];

    // analytical solution for temperature at fuel centerline
    // T(r=0) = T_outer + qdot / (4 * k) * (r_out^2 - r_in^2)
    double T_fuel_cl_analytical = 613.511111; // K

    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-6);

    std::vector<double> Tavg = solver.get_average_temperatures();
    EXPECT_EQ(Tavg.size(), 1);
    EXPECT_NEAR(Tavg[0], (T_fuel_cl + T_outer) / 2, 1e-6);
}

TEST(CylindricalSolverTest, TemperatureDistribution_FuelPin_3Regions) {
    double height = 0.11951; // m
    std::vector<double> radii = {0.0, 0.004096, 0.004180, 0.004750}; // fuel, gap, clad radii in m
    std::vector<CylinderNode> nodes = {
        CylinderNode(height, radii[0], radii[1]),
        CylinderNode(height, radii[1], radii[2]),
        CylinderNode(height, radii[2], radii[3])
    };

    double qdot_fuel = 1000 / nodes[0].get_volume(); // W/m^3

    CylindricalSolver solver(nodes);
    std::vector<double> k = {4.5, 0.2, 8.8};  // W/m-K
    std::vector<double> qdot = {qdot_fuel, 0.0, 0.0};  // W/m^3
    double T_outer = 600.0; // K

    std::vector<double> temps = solver.solve_temperatures(qdot, k, T_outer);
    double T_fuel_cl = temps[0];
    double T_fuel_outer = temps[1];
    double T_clad_inner = temps[2];
    double T_clad_outer = temps[3];

    // analytical solution for temperature at fuel centerline
    // T(r=0.) = T_outer + q * (R_fuel + R_gap + R_clad)
    // T(r=r1) = T_outer + q * (R_gap + R_clad)
    // T(r=r2) = T_outer + q * R_clad
    // T(r=r3) = T_outer
    double T_fuel_cl_analytical = 902.488178; // K
    double T_fuel_outer_analytical = 754.518279; // K
    double T_clad_inner_analytical = 619.345388; // K
    double T_clad_outer_analytical = 600.000000; // K

    EXPECT_NEAR(T_fuel_cl, T_fuel_cl_analytical, 1e-6);
    EXPECT_NEAR(T_fuel_outer, T_fuel_outer_analytical, 1e-6);
    EXPECT_NEAR(T_clad_inner, T_clad_inner_analytical, 1e-6);
    EXPECT_NEAR(T_clad_outer, T_clad_outer_analytical, 1e-6);

    std::vector<double> Tavg = solver.get_average_temperatures();
    EXPECT_EQ(Tavg.size(), 3);
    EXPECT_NEAR(Tavg[0], (T_fuel_cl + T_fuel_outer) / 2, 1e-6);
    EXPECT_NEAR(Tavg[1], (T_fuel_outer + T_clad_inner) / 2, 1e-6);
    EXPECT_NEAR(Tavg[2], (T_clad_inner + T_clad_outer) / 2, 1e-6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
