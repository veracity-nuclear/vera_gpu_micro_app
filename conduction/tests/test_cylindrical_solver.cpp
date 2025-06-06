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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
