#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>

#include "geometry.hpp"
#include "materials.hpp"
#include "solver.hpp"

TEST(SubchannelTest, OpenMP7x7Execution) {

    // geometric parameters
    size_t N = 10; // 10x10 grid
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    double length = 1.3e-2; // m
    size_t naxial = 25; // number of axial nodes to discretize to
    Geometry geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, N, naxial);

    // Explicitly use OpenMP execution space
    Water<Kokkos::OpenMP> fluid;

    Kokkos::initialize();
    {

        // Create views with OpenMP execution space
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

        std::cout << "Testing OpenMP execution space with 7x7 grid..." << std::endl;

        // Explicitly instantiate Solver with OpenMP
        Solver<Kokkos::OpenMP> solver(
            std::make_shared<Geometry>(geometry),
            std::make_shared<Water<Kokkos::OpenMP>>(fluid),
            inlet_temperature,
            inlet_pressure,
            linear_heat_rate,
            inlet_mass_flow
        );

        solver.solve();

        auto P = solver.get_surface_pressures();
        auto alpha = solver.get_surface_void_fractions();

        // Create host mirrors for accessing data
        auto h_P = Kokkos::create_mirror_view(P);
        auto h_alpha = Kokkos::create_mirror_view(alpha);

        Kokkos::deep_copy(h_P, P);
        Kokkos::deep_copy(h_alpha, alpha);

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
    Kokkos::finalize();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
