#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <Kokkos_Core.hpp>

#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"
#include "th.hpp"


class Solver {
public:
    using DoubleView1D = Kokkos::View<double*>;
    using DoubleView2D = Kokkos::View<double**>;

    Solver(
        std::shared_ptr<Geometry> geometry,
        std::shared_ptr<Water> fluid,
        DoubleView1D inlet_temperature,
        DoubleView1D inlet_pressure,
        DoubleView1D linear_heat_rate,
        DoubleView1D mass_flow_rate
    );
    ~Solver() = default;

    State state;

    void solve(size_t max_outer_iter = 10, size_t max_inner_iter = 10, bool debug = false);
    void print_state_at_plane(size_t k);

    DoubleView2D get_surface_liquid_enthalpies() const { return state.h_l; }
    DoubleView2D get_surface_temperatures() const { return state.fluid->T(state.h_l); }
    DoubleView2D get_surface_pressures() const { return state.P; }
    DoubleView2D get_surface_void_fractions() const { return state.alpha; }
    DoubleView2D get_surface_qualities() const { return state.X; }
    DoubleView2D get_evaporation_rates() const;
    DoubleView2D get_surface_liquid_flow_rates() const { return state.W_l; }
    DoubleView2D get_surface_vapor_flow_rates() const { return state.W_v; }
};
