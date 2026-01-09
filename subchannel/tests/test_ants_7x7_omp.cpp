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

TEST(SubchannelTest, 7x7_OpenMP) {

    // geometric parameters
    size_t N = 7; // 7x7 grid
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    double length = 1.3e-2; // m, length of axial momentum cell
    size_t naxial = 10; // number of axial nodes to discretize to

    // Create a core map for a single assembly (1x1)
    Kokkos::View<size_t**, Kokkos::OpenMP> core_map("core_map", 1, 1);
    core_map(0, 0) = 1;

    Geometry<Kokkos::OpenMP> geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, naxial, core_map);

    // create 1D views for each solver parameters
    Kokkos::View<double*, Kokkos::OpenMP> inlet_mass_flow("inlet_mass_flow", N*N);
    Kokkos::View<double*, Kokkos::OpenMP> inlet_temperature("inlet_temperature", N*N);
    Kokkos::View<double*, Kokkos::OpenMP> inlet_pressure("inlet_pressure", N*N);
    Kokkos::View<double*, Kokkos::OpenMP> linear_heat_rate("linear_heat_rate", N*N);

    auto h_inlet_mass_flow = Kokkos::create_mirror_view(inlet_mass_flow);
    auto h_inlet_temperature = Kokkos::create_mirror_view(inlet_temperature);
    auto h_inlet_pressure = Kokkos::create_mirror_view(inlet_pressure);
    auto h_linear_heat_rate = Kokkos::create_mirror_view(linear_heat_rate);

    // create a gradient heat rate distribution
    const double c_tl = 1.1, c_tr = 1.0, c_bl = 1.0, c_br = 0.9;
    for (int j = 0; j < N; ++j) {
        double v = double(j) / double(N - 1);
        for (int i = 0; i < N; ++i) {
            double u = double(i) / double(N - 1);
            double val =
                (1.0 - u) * (1.0 - v) * c_tl +
                u         * (1.0 - v) * c_tr +
                (1.0 - u) * v         * c_bl +
                u         * v         * c_br;
            h_linear_heat_rate[j * N + i] = val * 29.1e3; // W/m
        }
    }

    for (size_t i = 0; i < N*N; ++i) {
        h_inlet_mass_flow(i) = 0.25; // kg/s
        h_inlet_temperature(i) = 278.0 + 273.15; // K
        h_inlet_pressure(i) = 7.255e6; // Pa
    }

    Kokkos::deep_copy(inlet_mass_flow, h_inlet_mass_flow);
    Kokkos::deep_copy(inlet_temperature, h_inlet_temperature);
    Kokkos::deep_copy(inlet_pressure, h_inlet_pressure);
    Kokkos::deep_copy(linear_heat_rate, h_linear_heat_rate);

    Solver<Kokkos::OpenMP> solver(
        std::make_shared<Geometry<Kokkos::OpenMP>>(geometry),
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

    std::cout << "Exit Void Distribution" << std::endl;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            size_t k = naxial;
            std::cout << std::setw(12) << std::setprecision(3) << h_alpha(i + j*N, k) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Pressure Drop Distribution (kPa)" << std::endl;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            size_t k = naxial;
            std::cout << std::setw(12) << std::setprecision(6) << (h_P(i + j*N, 0) - h_P(i + j*N, k)) / 1000.0 << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
