#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "vectors.hpp"
#include "geometry.hpp"
#include "materials.hpp"
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

    Vector1D get_surface_enthalpies() const { return h; }
    Vector1D get_surface_temperatures() const { return fluid->T(h); }
    Vector1D get_surface_pressures() const { return P; }

private:
    double T_inlet;
    double P_inlet;
    double lhr;
    std::unique_ptr<Geometry> geom;
    std::unique_ptr<Water> fluid;

    // solution vectors
    Vector1D h; // enthalpy
    Vector1D P; // pressure
    Vector1D W_l, W_v; // liquid and vapor flow rates
    Vector1D alpha; // void fraction, for future implementation
};
