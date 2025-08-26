#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "constants.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "solver.hpp"

TEST(SubchannelTest, SingleChannel) {

    // geometric parameters
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    size_t naxial = 10; // number of axial nodes to discretize to
    Geometry geometry(height, flow_area, hydraulic_diameter, gap_width, naxial);

    // working fluid is water
    Water fluid;

    // solver parameters
    double inlet_mass_flow = 2.25; // kg/s
    double inlet_temperature = 278.0 + 273.15; // K
    double inlet_pressure = 7.255e6; // Pa
    double linear_heat_rate = 29.1; // W/m (0.1% of BWR case to prevent boiling for now)

    Solver solver(
        std::make_unique<Geometry>(geometry),
        std::make_unique<Water>(fluid),
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );
    solver.solve();

    std::vector<double> h = solver.get_surface_enthalpies();
    std::vector<double> T = solver.get_surface_temperatures();

    // Print table of results
    std::cout << std::fixed << std::setprecision(2) << std::endl;
    std::cout << "Subchannel: 0" << std::endl;
    std::cout << std::setw(6) << "Surf" << std::setw(12) << "Enthalpy" << std::setw(12) << "Temp." << std::endl;
    std::cout << std::setw(6) << "" << std::setw(12) << "(J/kg)" << std::setw(12) << "(K)" << std::endl;
    for (size_t k = 0; k < naxial + 1; ++k) {
        std::cout << std::setw(6) << k << std::setw(12) << h[k] << std::setw(12) << fluid.T(h[k]) << std::endl;
    }
    std::cout << std::endl;

    double total_heat = linear_heat_rate * geometry.height();
    double expected_deltaT = total_heat / (inlet_mass_flow * fluid.Cp(fluid.h(inlet_temperature)));
    double actual_deltaT = T.back() - T.front();

    // check total temperature rise in subchannel
    EXPECT_NEAR(actual_deltaT, expected_deltaT, 1e-6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
