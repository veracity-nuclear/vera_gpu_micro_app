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
{
    // initialize solution vectors
    state.fluid = *this->fluid; // set fluid reference in state
    state.h_l.resize(geom->naxial() + 1, fluid->h(T_inlet));
    state.h_v.resize(geom->naxial() + 1, fluid->h(T_inlet));
    state.P.resize(geom->naxial() + 1, P_inlet);
    state.W_l.resize(geom->naxial() + 1, mass_flow_rate);
    state.W_v.resize(geom->naxial() + 1, 0.0);
    state.alpha.resize(geom->naxial() + 1, 0.0);
    state.X.resize(geom->naxial() + 1, 0.0);
    state.lhr.resize(geom->naxial(), linear_heat_rate);
    state.evap.resize(geom->naxial(), 0.0);

    std::cout << "Solver initialized." << std::endl;
}

Vector1D Solver::get_evaporation_rates() const {
    Vector1D evap_rates = state.evap;
    for (double& rate : evap_rates) {
        rate *= geom->dz(); // convert from [kg/m/s] to [kg/s]
    }
    return evap_rates;
}

void Solver::solve() {
    for (size_t iter = 0; iter < 1000; ++iter) { // fixed number of iterations for now
        TH::solve_evaporation_term(state, *geom, *fluid);
        TH::solve_turbulent_mixing(state, *geom, *fluid);   // --- PLACEHOLDER FOR NOW ---
        TH::solve_void_drift(state, *geom, *fluid);         // --- PLACEHOLDER FOR NOW ---
        TH::solve_surface_mass_flux(state, *geom, *fluid);  // --- PLACEHOLDER FOR NOW ---
        TH::solve_flow_rates(state, *geom, *fluid);
        TH::solve_enthalpy(state, *geom, *fluid);
        TH::solve_void_fraction(state, *geom, *fluid);
        TH::solve_quality(state, *geom, *fluid);
        TH::solve_pressure(state, *geom, *fluid);
    }
}
