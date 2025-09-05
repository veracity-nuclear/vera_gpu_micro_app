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
    h.resize(geom->naxial() + 1, fluid->h(T_inlet));
    P.resize(geom->naxial() + 1, P_inlet);
    W_l.resize(geom->naxial() + 1, mass_flow_rate);
    W_v.resize(geom->naxial() + 1, 0.0);
    alpha.resize(geom->naxial() + 1, 0.0);

    std::cout << "Solver initialized." << std::endl;
}

void Solver::solve() {
    TH::solve_enthalpy(h, W_l, lhr, *geom);
    TH::solve_pressure(P, W_l, fluid->rho(h), fluid->mu(h), *geom, *fluid);
}
