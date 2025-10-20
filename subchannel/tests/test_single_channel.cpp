#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include "vectors.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "solver.hpp"

TEST(SubchannelTest, SingleChannel) {

    // geometric parameters
    size_t N = 1; // number of subchannels in each direction
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    size_t naxial = 10; // number of axial nodes to discretize to
    Geometry geometry(height, flow_area, hydraulic_diameter, gap_width, N, N, naxial);

    // working fluid is water
    Water fluid;

    // create 2D array for each solver parameters
    Vector2D inlet_mass_flow(N, Vector1D(N, 2.25 / 9)); // kg/s
    Vector2D inlet_temperature(N, Vector1D(N, 278.0 + 273.15)); // K
    Vector2D inlet_pressure(N, Vector1D(N, 7.255e6)); // Pa
    Vector2D linear_heat_rate(N, Vector1D(N, 29.1e3)); // W/m

    std::cout << "Linear heat rate: " << linear_heat_rate[0][0] << " W/m" << std::endl;

    Solver solver(
        std::make_unique<Geometry>(geometry),
        std::make_unique<Water>(fluid),
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );
    solver.solve();

    Vector3D h = solver.get_surface_liquid_enthalpies();
    Vector3D T = solver.get_surface_temperatures();
    Vector3D P = solver.get_surface_pressures();
    Vector3D alpha = solver.get_surface_void_fractions();
    Vector3D X = solver.get_surface_qualities();
    Vector3D evap = solver.get_evaporation_rates();
    Vector3D W_l = solver.get_surface_liquid_flow_rates();
    Vector3D W_v = solver.get_surface_vapor_flow_rates();

    // Print table of results
    std::cout << std::fixed << std::setprecision(2) << std::endl;
    std::cout << "Subchannel: 0" << std::endl;
    std::cout << std::setw(6) << "Surf"
              << std::setw(12) << "Enthalpy"
              << std::setw(12) << "Temp."
              << std::setw(12) << "Press."
              << std::setw(12) << "Alpha"
              << std::setw(12) << "Quality"
              << std::setw(12) << "Liq. MFR"
              << std::setw(12) << "Vap. MFR"
              << std::endl;

    std::cout << std::setw(6) << ""
              << std::setw(12) << "(kJ/kg)"
              << std::setw(12) << "(K)"
              << std::setw(12) << "(kPa)"
              << std::setw(12) << "(-)"
              << std::setw(12) << "(-)"
              << std::setw(12) << "(kg/s)"
              << std::setw(12) << "(kg/s)"
              << std::endl;

    for (size_t k = 0; k < naxial + 1; ++k) {
        std::cout << std::setw(6)  << std::setprecision(2) << k
                  << std::setw(12) << std::setprecision(2) << h[0][0][k] / 1000.0
                  << std::setw(12) << std::setprecision(2) << fluid.T(h[0][0][k])
                  << std::setw(12) << std::setprecision(2) << P[0][0][k] / 1000.0
                  << std::setw(12) << std::setprecision(3) << alpha[0][0][k]
                  << std::setw(12) << std::setprecision(3) << X[0][0][k]
                  << std::setw(12) << std::setprecision(3) << W_l[0][0][k]
                  << std::setw(12) << std::setprecision(3) << W_v[0][0][k]
                  << std::endl;
    }
    std::cout << std::endl;

    double expected_deltaT = 21.729682; // expected temperature rise in subchannel, K
    double actual_deltaT = T[0][0].back() - T[0][0].front();
    std::cout << "Total temperature rise: " << actual_deltaT << " K" << std::endl;

    // check total temperature rise in subchannel
    EXPECT_NEAR(actual_deltaT, expected_deltaT, 1e-6);

    double total_pressure_drop = P[0][0].front() - P[0][0].back();
    double expected_pressure_drop = 73058.774; // expected pressure drop in subchannel, Pa
    std::cout << "Total pressure drop: " << total_pressure_drop / 1000.0 << " kPa" << std::endl;

    // check total pressure drop in subchannel
    EXPECT_NEAR(total_pressure_drop, expected_pressure_drop, 1e-3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
