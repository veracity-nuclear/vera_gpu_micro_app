#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>

#include "argument_parser.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "solver.hpp"

TEST(SubchannelTest, SMR_Serial) {

    const char* raw_args[] = {
        "exe",
        "../subchannel/data/HT1C1_dep.h5",
        "--device", "serial",
        "--no-crossflow"
    };
    char** args = const_cast<char**>(raw_args);
    int argc = 5;

    // Initialize Kokkos
    Kokkos::initialize(argc, args);
    {

        auto parser = ArgumentParser::vera_gpu_subchannel_parser(raw_args[0]);
        parser.parse(argc, args);
        Solver<Kokkos::Serial> solver(parser);
        solver.solve();

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
        auto h_lhr = Kokkos::create_mirror_view(solver.state.lhr);

        Kokkos::deep_copy(h_h, h);
        Kokkos::deep_copy(h_T, T);
        Kokkos::deep_copy(h_P, P);
        Kokkos::deep_copy(h_alpha, alpha);
        Kokkos::deep_copy(h_X, X);
        Kokkos::deep_copy(h_evap, evap);
        Kokkos::deep_copy(h_W_l, W_l);
        Kokkos::deep_copy(h_W_v, W_v);
        Kokkos::deep_copy(h_lhr, solver.state.lhr);

        size_t ai = 5;
        size_t aj = 5;

        size_t k = solver.state.surface_plane;
        // size_t k = solver.state.geom->naxial() - 1;

        std::cout << "Exit Void Distribution" << std::endl;
        for (size_t j = 0; j < solver.state.geom->nchan(); ++j) {
            for (size_t i = 0; i < solver.state.geom->nchan(); ++i) {
                size_t aij = solver.state.geom->global_chan_index(aj, ai, j, i);
                std::cout << std::setw(12) << std::setprecision(3) << h_alpha(aij, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Exit Pressure Distribution" << std::endl;
        for (size_t j = 0; j < solver.state.geom->nchan(); ++j) {
            for (size_t i = 0; i < solver.state.geom->nchan(); ++i) {
                size_t aij = solver.state.geom->global_chan_index(aj, ai, j, i);
                std::cout << std::setw(12) << std::setprecision(3) << h_P(aij, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        // Midplane linear heat rate distribution
        size_t k_mid = solver.state.geom->naxial() / 2;
        std::cout << "Midplane (k=" << k_mid << ") Linear Heat Rate Distribution [W/m]" << std::endl;
        for (size_t j = 0; j < solver.state.geom->nchan(); ++j) {
            for (size_t i = 0; i < solver.state.geom->nchan(); ++i) {
                size_t aij = solver.state.geom->global_chan_index(aj, ai, j, i);
                std::cout << std::setw(12) << std::setprecision(3) << h_lhr(aij, k_mid) << " ";
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
