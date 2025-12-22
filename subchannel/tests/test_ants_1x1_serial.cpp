#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>

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
    double length = 1.3e-2; // m, length of axial momentum cell
    size_t naxial = 10; // number of axial nodes to discretize to
    Geometry<Kokkos::Serial> geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, naxial);

    // working fluid is water
    Water<Kokkos::Serial> fluid;

    // create 1D views for each solver parameters
    Kokkos::View<double*, Kokkos::Serial> inlet_mass_flow("inlet_mass_flow", N*N);
    Kokkos::View<double*, Kokkos::Serial> inlet_temperature("inlet_temperature", N*N);
    Kokkos::View<double*, Kokkos::Serial> inlet_pressure("inlet_pressure", N*N);
    Kokkos::View<double*, Kokkos::Serial> linear_heat_rate("linear_heat_rate", N*N);

    auto h_inlet_mass_flow = Kokkos::create_mirror_view(inlet_mass_flow);
    auto h_inlet_temperature = Kokkos::create_mirror_view(inlet_temperature);
    auto h_inlet_pressure = Kokkos::create_mirror_view(inlet_pressure);
    auto h_linear_heat_rate = Kokkos::create_mirror_view(linear_heat_rate);

    for (size_t i = 0; i < N*N; ++i) {
        h_inlet_mass_flow(i) = 0.25; // kg/s
        h_inlet_temperature(i) = 278.0 + 273.15; // K
        h_inlet_pressure(i) = 7.255e6; // Pa
        h_linear_heat_rate(i) = 29.1e3; // W/m
    }

    Kokkos::deep_copy(inlet_mass_flow, h_inlet_mass_flow);
    Kokkos::deep_copy(inlet_temperature, h_inlet_temperature);
    Kokkos::deep_copy(inlet_pressure, h_inlet_pressure);
    Kokkos::deep_copy(linear_heat_rate, h_linear_heat_rate);

    std::cout << "Linear heat rate: " << h_linear_heat_rate(0) << " W/m" << std::endl;

    Solver<Kokkos::Serial> solver(
        std::make_shared<Geometry<Kokkos::Serial>>(geometry),
        std::make_shared<Water<Kokkos::Serial>>(fluid),
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );
    solver.solve();

    auto h = solver.get_surface_liquid_enthalpies();
    auto P = solver.get_surface_pressures();
    auto alpha = solver.get_surface_void_fractions();
    auto X = solver.get_surface_qualities();
    auto evap = solver.get_evaporation_rates();
    auto W_l = solver.get_surface_liquid_flow_rates();
    auto W_v = solver.get_surface_vapor_flow_rates();

    // Create host mirrors for accessing data
    auto h_h = Kokkos::create_mirror_view(h);
    auto h_P = Kokkos::create_mirror_view(P);
    auto h_alpha = Kokkos::create_mirror_view(alpha);
    auto h_X = Kokkos::create_mirror_view(X);
    auto h_evap = Kokkos::create_mirror_view(evap);
    auto h_W_l = Kokkos::create_mirror_view(W_l);
    auto h_W_v = Kokkos::create_mirror_view(W_v);

    Kokkos::deep_copy(h_h, h);
    Kokkos::deep_copy(h_P, P);
    Kokkos::deep_copy(h_alpha, alpha);
    Kokkos::deep_copy(h_X, X);
    Kokkos::deep_copy(h_evap, evap);
    Kokkos::deep_copy(h_W_l, W_l);
    Kokkos::deep_copy(h_W_v, W_v);

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
                << std::setw(12) << std::setprecision(2) << h_h(0, k) / 1000.0
                << std::setw(12) << std::setprecision(2) << fluid.T(h_h(0, k))
                << std::setw(12) << std::setprecision(2) << h_P(0, k) / 1000.0
                << std::setw(12) << std::setprecision(3) << h_alpha(0, k)
                << std::setw(12) << std::setprecision(3) << h_X(0, k)
                << std::setw(12) << std::setprecision(3) << h_W_l(0, k)
                << std::setw(12) << std::setprecision(3) << h_W_v(0, k)
                << std::endl;
    }
    std::cout << std::endl;

    double expected_deltaT = 21.728303; // expected temperature rise in subchannel, K
    double actual_deltaT = fluid.T(h_h(0, naxial)) - fluid.T(h_h(0, 0));
    std::cout << "Total temperature rise: " << actual_deltaT << " K" << std::endl;

    // check total temperature rise in subchannel
    EXPECT_NEAR(actual_deltaT, expected_deltaT, 1e-6);

    double total_pressure_drop = h_P(0, 0) - h_P(0, naxial);
    double expected_pressure_drop = 86176.795; // expected pressure drop in subchannel, Pa
    std::cout << "Total pressure drop: " << total_pressure_drop / 1000.0 << " kPa" << std::endl;

    // check total pressure drop in subchannel
    EXPECT_NEAR(total_pressure_drop, expected_pressure_drop, 1e-3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
