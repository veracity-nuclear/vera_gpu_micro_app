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
    Geometry geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, N, naxial);

    // working fluid is water
    Water fluid;

    // create 2D array for each solver parameters
    Vector1D inlet_mass_flow(N*N, 2.25 / (N * N)); // kg/s
    Vector1D inlet_temperature(N*N, 278.0 + 273.15); // K
    Vector1D inlet_pressure(N*N, 7.255e6); // Pa
    Vector1D linear_heat_rate(N*N, 29.1e3); // W/m

    linear_heat_rate[4] = 0.0; // no power in center subchannel

    Solver solver(
        std::make_shared<Geometry>(geometry),
        std::make_shared<Water>(fluid),
        inlet_temperature,
        inlet_pressure,
        linear_heat_rate,
        inlet_mass_flow
    );

    size_t outer_iter = 1;
    size_t inner_iter = 1;
    solver.solve(outer_iter, inner_iter);

    Vector2D h = solver.get_surface_liquid_enthalpies();
    Vector2D T = solver.get_surface_temperatures();
    Vector2D P = solver.get_surface_pressures();
    Vector2D alpha = solver.get_surface_void_fractions();
    Vector2D X = solver.get_surface_qualities();
    Vector2D evap = solver.get_evaporation_rates();
    Vector2D W_l = solver.get_surface_liquid_flow_rates();
    Vector2D W_v = solver.get_surface_vapor_flow_rates();

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
                        << std::setw(12) << std::setprecision(2) << h[i][k] / 1000.0
                        << std::setw(12) << std::setprecision(2) << fluid.T(h[i][k])
                        << std::setw(12) << std::setprecision(2) << P[i][k] / 1000.0
                        << std::setw(12) << std::setprecision(3) << alpha[i][k]
                        << std::setw(12) << std::setprecision(3) << X[i][k]
                        << std::setw(12) << std::setprecision(3) << W_l[i][k]
                        << std::setw(12) << std::setprecision(3) << W_v[i][k]
                        << std::endl;
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    // // print exit plane data to compare to ANTS Theory results
    // std::cout << "Exit Void Distribution" << std::endl;
    // for (size_t j = 0; j < N; ++j) {
    //     for (size_t i = 0; i < N; ++i) {
    //         size_t k = naxial;
    //         std::cout << std::setw(12) << std::setprecision(3) << alpha[i + j*N][k] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // Vector2D ants_void({
    //     {0.808745, 0.785275, 0.808745},
    //     {0.785275, 0.695971, 0.785275},
    //     {0.808745, 0.785275, 0.808745}
    // });

    // std::cout << "Exit Void Distribution Error vs. ANTS" << std::endl;
    // double max_void_error = 0.0;
    // for (size_t j = 0; j < N; ++j) {
    //     for (size_t i = 0; i < N; ++i) {
    //         size_t k = naxial;
    //         double void_error = std::abs((alpha[i + j*N][k] - ants_void[i][j]) / ants_void[i][j]);
    //         if (void_error > max_void_error) {
    //             max_void_error = void_error;
    //         }
    //         std::cout << std::setw(12) << std::setprecision(3) << (alpha[i + j*N][k] - ants_void[i][j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // std::cout << "Maximum Void Error: " << std::setw(8) << std::setprecision(6) << max_void_error * 100.0 << " %" << std::endl;
    // std::cout << std::endl;

    // // print exit plane data to compare to ANTS Theory results
    // std::cout << "Pressure Drop Distribution (kPa)" << std::endl;
    // for (size_t j = 0; j < N; ++j) {
    //     for (size_t i = 0; i < N; ++i) {
    //         size_t k = naxial;
    //         std::cout << std::setw(12) << std::setprecision(6) << (P[i + j*N][0] - P[i + j*N][k]) / 1000.0 << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // Vector2D ants_pressure_drop({
    //     {81.685284, 81.685286, 81.685284},
    //     {81.685286, 81.685310, 81.685286},
    //     {81.685284, 81.685286, 81.685284}
    // });

    // std::cout << "Pressure Drop Distribution Error vs. ANTS (kPa)" << std::endl;
    // double max_pressure_drop_error = 0.0;
    // for (size_t j = 0; j < N; ++j) {
    //     for (size_t i = 0; i < N; ++i) {
    //         size_t k = naxial;
    //         double pressure_drop_error = std::abs((P[i + j*N][0] - P[i + j*N][k]) / 1000.0 - ants_pressure_drop[i][j]) / ants_pressure_drop[i][j];
    //         if (pressure_drop_error > max_pressure_drop_error) {
    //             max_pressure_drop_error = pressure_drop_error;
    //         }
    //         std::cout << std::setw(12) << std::setprecision(6) << ((P[i + j*N][0] - P[i + j*N][k])) / 1000.0 - ants_pressure_drop[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // std::cout << "Maximum Pressure Drop Error: " << std::setw(8) << std::setprecision(6) << max_pressure_drop_error * 100.0 << " %" << std::endl;
    // std::cout << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
