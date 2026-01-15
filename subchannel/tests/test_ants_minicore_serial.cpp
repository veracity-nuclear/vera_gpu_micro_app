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
    size_t N = 10; // NxN pins in assembly
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
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );

    size_t outer_iter = 25;
    size_t inner_iter = 50;
    solver.solve(outer_iter, inner_iter);
    solver.print_state_at_plane(solver.state.surface_plane);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
