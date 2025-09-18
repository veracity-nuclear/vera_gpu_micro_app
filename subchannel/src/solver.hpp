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
        std::unique_ptr<Geometry> geometry,
        std::unique_ptr<Water> fluid,
        double inlet_temperature,
        double inlet_pressure,
        double linear_heat_rate,
        double mass_flow_rate
    );
    ~Solver() = default;

    void solve();

    Vector1D get_surface_liquid_enthalpies() const { return state.h_l; }
    Vector1D get_surface_temperatures() const { return fluid->T(state.h_l); }
    Vector1D get_surface_pressures() const { return state.P; }
    Vector1D get_surface_void_fractions() const { return state.alpha; }
    Vector1D get_surface_qualities() const { return state.X; }
    Vector1D get_evaporation_rates() const;
    Vector1D get_surface_liquid_flow_rates() const { return state.W_l; }
    Vector1D get_surface_vapor_flow_rates() const { return state.W_v; }

private:
    double T_inlet;
    double P_inlet;
    double lhr;
    std::unique_ptr<Geometry> geom;
    std::unique_ptr<Water> fluid;
    State state;
};
