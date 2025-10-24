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
        Vector2D inlet_temperature,
        Vector2D inlet_pressure,
        Vector2D linear_heat_rate,
        Vector2D mass_flow_rate
    );
    ~Solver() = default;

    void solve();

    Vector3D get_surface_liquid_enthalpies() const { return state.h_l; }
    Vector3D get_surface_temperatures() const { return state.fluid->T(state.h_l); }
    Vector3D get_surface_pressures() const { return state.P; }
    Vector3D get_surface_void_fractions() const { return state.alpha; }
    Vector3D get_surface_qualities() const { return state.X; }
    Vector3D get_evaporation_rates() const;
    Vector3D get_surface_liquid_flow_rates() const { return state.W_l; }
    Vector3D get_surface_vapor_flow_rates() const { return state.W_v; }

private:
    Vector2D T_inlet;
    Vector2D P_inlet;
    Vector2D lhr;
    State state;
};
