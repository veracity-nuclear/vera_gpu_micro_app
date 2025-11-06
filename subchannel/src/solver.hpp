#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "vectors.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"
#include "th.hpp"


class Solver {
public:
    Solver(
        std::shared_ptr<Geometry> geometry,
        std::shared_ptr<Water> fluid,
        Vector1D inlet_temperature,
        Vector1D inlet_pressure,
        Vector1D linear_heat_rate,
        Vector1D mass_flow_rate
    );
    ~Solver() = default;

    State state;

    void solve(size_t max_outer_iter = 10, size_t max_inner_iter = 10);

    Vector2D get_surface_liquid_enthalpies() const { return state.h_l; }
    Vector2D get_surface_temperatures() const { return state.fluid->T(state.h_l); }
    Vector2D get_surface_pressures() const { return state.P; }
    Vector2D get_surface_void_fractions() const { return state.alpha; }
    Vector2D get_surface_qualities() const { return state.X; }
    Vector2D get_evaporation_rates() const;
    Vector2D get_surface_liquid_flow_rates() const { return state.W_l; }
    Vector2D get_surface_vapor_flow_rates() const { return state.W_v; }
};
