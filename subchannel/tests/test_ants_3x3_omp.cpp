#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>

#include "geometry.hpp"
#include "materials.hpp"
#include "solver.hpp"

TEST(SubchannelTest, OpenMPExecution) {

    // geometric parameters
    size_t N = 3; // 3x3 grid
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    double length = 1.3e-2; // m
    size_t naxial = 50;

    // Create a core map for a single assembly (1x1)
    Kokkos::View<size_t**, Kokkos::OpenMP> core_map("core_map", 1, 1);
    core_map(0, 0) = 1;

    Geometry<Kokkos::OpenMP> geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, naxial, core_map);

    // Explicitly use OpenMP execution space
    Water<Kokkos::OpenMP> fluid;

    // Create views with OpenMP execution space
    Kokkos::View<double*, Kokkos::OpenMP> inlet_mass_flow("inlet_mass_flow", N*N);
    Kokkos::View<double*, Kokkos::OpenMP> inlet_temperature("inlet_temperature", N*N);
    Kokkos::View<double*, Kokkos::OpenMP> inlet_pressure("inlet_pressure", N*N);
    Kokkos::View<double*, Kokkos::OpenMP> linear_heat_rate("linear_heat_rate", N*N);

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

    std::cout << "Testing OpenMP execution space..." << std::endl;

    // Explicitly instantiate Solver with OpenMP
    Solver<Kokkos::OpenMP> solver(
        std::make_shared<Geometry<Kokkos::OpenMP>>(geometry),
        std::make_shared<Water<Kokkos::OpenMP>>(fluid),
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );

    solver.solve();

    auto P = solver.get_surface_pressures();
    auto h_P = Kokkos::create_mirror_view(P);
    Kokkos::deep_copy(h_P, P);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
