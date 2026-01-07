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
#include "linear_algebra.hpp"

TEST(SubchannelTest, 3x3Channels) {

    // geometric parameters
    size_t N = 3; // number of subchannels in each direction
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    double length = 1.3e-2; // m, length of axial momentum cell
    size_t naxial = 10; // number of axial nodes to discretize to

    // Create a core map for a single assembly (1x1)
    Kokkos::View<size_t**, Kokkos::Serial> core_map("core_map", 1, 1);
    core_map(0, 0) = 1;

    Geometry<Kokkos::Serial> geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, naxial, core_map);

    // working fluid is water
    Water fluid;

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
        h_inlet_mass_flow(i) = 2.25 / (N * N); // kg/s
        h_inlet_temperature(i) = 278.0 + 273.15; // K
        h_inlet_pressure(i) = 7.255e6; // Pa
        h_linear_heat_rate(i) = 29.1e3; // W/m
    }

    h_linear_heat_rate(4) = 0.0; // no power in center subchannel

    Kokkos::deep_copy(inlet_mass_flow, h_inlet_mass_flow);
    Kokkos::deep_copy(inlet_temperature, h_inlet_temperature);
    Kokkos::deep_copy(inlet_pressure, h_inlet_pressure);
    Kokkos::deep_copy(linear_heat_rate, h_linear_heat_rate);

    Solver<Kokkos::Serial> solver(
        std::make_shared<Geometry<Kokkos::Serial>>(geometry),
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );

    size_t outer_iter = 100;
    size_t inner_iter = 100;
    solver.solve(outer_iter, inner_iter);

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

    // activate to print all subchannel table data
    if (false) {
        for (size_t i = 0; i < N*N; ++i) {
            std::cout << std::fixed << std::setprecision(2) << std::endl;
            std::cout << "Subchannel: " << i << std::endl;
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
                        << std::setw(12) << std::setprecision(2) << h_h(i, k) / 1000.0
                        << std::setw(12) << std::setprecision(2) << fluid.T(h_h(i, k))
                        << std::setw(12) << std::setprecision(2) << h_P(i, k) / 1000.0
                        << std::setw(12) << std::setprecision(3) << h_alpha(i, k)
                        << std::setw(12) << std::setprecision(3) << h_X(i, k)
                        << std::setw(12) << std::setprecision(3) << h_W_l(i, k)
                        << std::setw(12) << std::setprecision(3) << h_W_v(i, k)
                        << std::endl;
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    // print exit plane data to compare to ANTS Theory results
    std::cout << "Exit Void Distribution" << std::endl;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            size_t k = naxial;
            std::cout << std::setw(12) << std::setprecision(3) << h_alpha(i + j*N, k) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::vector<std::vector<double>> ants_void = {
        {0.808745, 0.785275, 0.808745},
        {0.785275, 0.695971, 0.785275},
        {0.808745, 0.785275, 0.808745}
    };

    std::cout << "Exit Void Distribution Error vs. ANTS" << std::endl;
    double max_void_error = 0.0;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            size_t k = naxial;
            double void_error = std::abs((h_alpha(i + j*N, k) - ants_void[i][j]) / ants_void[i][j]);
            if (void_error > max_void_error) {
                max_void_error = void_error;
            }
            std::cout << std::setw(12) << std::setprecision(3) << (h_alpha(i + j*N, k) - ants_void[i][j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Maximum Void Error: " << std::setw(8) << std::setprecision(6) << max_void_error * 100.0 << " %" << std::endl;
    std::cout << std::endl;

    // print exit plane data to compare to ANTS Theory results
    std::cout << "Pressure Drop Distribution (kPa)" << std::endl;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            size_t k = naxial;
            std::cout << std::setw(12) << std::setprecision(6) << (h_P(i + j*N, 0) - h_P(i + j*N, k)) / 1000.0 << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::vector<std::vector<double>> ants_pressure_drop = {
        {81.685284, 81.685286, 81.685284},
        {81.685286, 81.685310, 81.685286},
        {81.685284, 81.685286, 81.685284}
    };

    std::cout << "Pressure Drop Distribution Error vs. ANTS (kPa)" << std::endl;
    double max_pressure_drop_error = 0.0;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            size_t k = naxial;
            double pressure_drop_error = std::abs((h_P(i + j*N, 0) - h_P(i + j*N, k)) / 1000.0 - ants_pressure_drop[i][j]) / ants_pressure_drop[i][j];
            if (pressure_drop_error > max_pressure_drop_error) {
                max_pressure_drop_error = pressure_drop_error;
            }
            std::cout << std::setw(12) << std::setprecision(6) << ((h_P(i + j*N, 0) - h_P(i + j*N, k))) / 1000.0 - ants_pressure_drop[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Maximum Pressure Drop Error: " << std::setw(8) << std::setprecision(6) << max_pressure_drop_error * 100.0 << " %" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
