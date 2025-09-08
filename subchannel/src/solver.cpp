#include "solver.hpp"

Solver::Solver(
    std::unique_ptr<Geometry> geometry,
    std::unique_ptr<Water> fluid,
    double inlet_temperature,
    double inlet_pressure,
    double linear_heat_rate,
    double mass_flow_rate
)
    : geom(std::move(geometry))
    , fluid(std::move(fluid))
    , T_inlet(inlet_temperature)
    , P_inlet(inlet_pressure)
    , lhr(linear_heat_rate)
{
    // initialize solution vectors
    state.h.resize(geom->naxial() + 1, fluid->h(T_inlet));
    state.P.resize(geom->naxial() + 1, P_inlet);
    state.W_l.resize(geom->naxial() + 1, mass_flow_rate);
    state.W_v.resize(geom->naxial() + 1, 0.0);
    state.alpha.resize(geom->naxial() + 1, 0.0);

    std::cout << "Solver initialized." << std::endl;
}

void Solver::solve() {
    TH::solve_enthalpy(state, lhr, *geom);
    TH::solve_pressure(state, *geom, *fluid);
}
