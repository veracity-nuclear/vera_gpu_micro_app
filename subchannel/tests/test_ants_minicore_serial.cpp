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

TEST(SubchannelTest, Minicore_Serial) {

    // geometric parameters
    size_t N = 5; // 5x5 grid
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    double length = 1.3e-2; // m, length of axial momentum cell
    size_t naxial = 10; // number of axial nodes to discretize to

    // Create a core map
    std::vector<std::vector<size_t>> map = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 0}
    };
    Kokkos::View<size_t**, Kokkos::Serial> core_map("core_map", map.size(), map[0].size());
    for (size_t aj = 0; aj < map.size(); ++aj) {
        for (size_t ai = 0; ai < map[aj].size(); ++ai) {
            core_map(aj, ai) = map[aj][ai];
        }
    }

    Geometry<Kokkos::Serial> geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, naxial, core_map);

    // working fluid is water
    Water<Kokkos::Serial> fluid;

    // create 1D views for each solver parameters
    Kokkos::View<double*, Kokkos::Serial> inlet_mass_flow("inlet_mass_flow", geometry.nchannels());
    Kokkos::View<double*, Kokkos::Serial> inlet_temperature("inlet_temperature", geometry.nchannels());
    Kokkos::View<double*, Kokkos::Serial> inlet_pressure("inlet_pressure", geometry.nchannels());
    Kokkos::View<double*, Kokkos::Serial> linear_heat_rate("linear_heat_rate", geometry.nchannels());

    auto h_inlet_mass_flow = Kokkos::create_mirror_view(inlet_mass_flow);
    auto h_inlet_temperature = Kokkos::create_mirror_view(inlet_temperature);
    auto h_inlet_pressure = Kokkos::create_mirror_view(inlet_pressure);
    auto h_linear_heat_rate = Kokkos::create_mirror_view(linear_heat_rate);

    // create a gradient heat rate distribution
    const double c_tl = 1.1, c_tr = 1.0, c_bl = 1.0, c_br = 0.9;
    for (size_t aj = 0; aj < core_map.extent(0); ++aj) {
        for (size_t ai = 0; ai < core_map.extent(1); ++ai) {
            if (core_map(aj, ai) == 0) continue; // skip non-existent assemblies
            for (int j = 0; j < N; ++j) {
                double v = double(j) / double(N - 1);
                for (int i = 0; i < N; ++i) {
                    size_t aij = geometry.global_chan_index(aj, ai, j, i);
                    double u = double(i) / double(N - 1);
                    double val =
                        (1.0 - u) * (1.0 - v) * c_tl +
                        u         * (1.0 - v) * c_tr +
                        (1.0 - u) * v         * c_bl +
                        u         * v         * c_br;
                    h_linear_heat_rate[aij] = val * 29.1e3; // W/m
                }
            }
        }
    }

    for (size_t i = 0; i < geometry.nchannels(); ++i) {
        h_inlet_mass_flow(i) = 0.25; // kg/s
        h_inlet_temperature(i) = 278.0 + 273.15; // K
        h_inlet_pressure(i) = 7.255e6; // Pa
    }

    Kokkos::deep_copy(inlet_mass_flow, h_inlet_mass_flow);
    Kokkos::deep_copy(inlet_temperature, h_inlet_temperature);
    Kokkos::deep_copy(inlet_pressure, h_inlet_pressure);
    Kokkos::deep_copy(linear_heat_rate, h_linear_heat_rate);

    Solver<Kokkos::Serial> solver(
        std::make_shared<Geometry<Kokkos::Serial>>(geometry),
        std::make_shared<Water<Kokkos::Serial>>(fluid),
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );

    size_t outer_iter = 100;
    size_t inner_iter = 100;
    solver.solve(outer_iter, inner_iter);

    auto h = solver.get_surface_liquid_enthalpies();
    auto T = solver.get_surface_temperatures();
    auto P = solver.get_surface_pressures();
    auto alpha = solver.get_surface_void_fractions();
    auto X = solver.get_surface_qualities();
    auto evap = solver.get_evaporation_rates();
    auto W_l = solver.get_surface_liquid_flow_rates();
    auto W_v = solver.get_surface_vapor_flow_rates();

    // Create host mirrors for accessing data
    auto h_h = Kokkos::create_mirror_view(h);
    auto h_T = Kokkos::create_mirror_view(T);
    auto h_P = Kokkos::create_mirror_view(P);
    auto h_alpha = Kokkos::create_mirror_view(alpha);
    auto h_X = Kokkos::create_mirror_view(X);
    auto h_evap = Kokkos::create_mirror_view(evap);
    auto h_W_l = Kokkos::create_mirror_view(W_l);
    auto h_W_v = Kokkos::create_mirror_view(W_v);

    Kokkos::deep_copy(h_h, h);
    Kokkos::deep_copy(h_T, T);
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

    solver.print_state_at_plane(solver.state.surface_plane);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
