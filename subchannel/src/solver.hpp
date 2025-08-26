#pragma once

#include <iostream>
#include <memory>
#include <vector>
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

    std::vector<double> get_surface_enthalpies() const { return h; }
    std::vector<double> get_surface_temperatures() const { return fluid->T(h); }

private:
    double T_inlet;
    double P_inlet;
    double lhr;
    std::unique_ptr<Geometry> geom;
    std::unique_ptr<Water> fluid;

    // solution vectors
    std::vector<double> h; // enthalpy
    std::vector<double> P; // pressure
    std::vector<double> W_l, W_v; // liquid and vapor flow rates
    std::vector<double> alpha; // void fraction, for future implementation
};
