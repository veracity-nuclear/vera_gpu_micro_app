#include "solver.hpp"

Solver::Solver(
    std::unique_ptr<Geometry> geometry,
    std::unique_ptr<Water> fluid,
    Vector2D inlet_temperature,
    Vector2D inlet_pressure,
    Vector2D linear_heat_rate,
    Vector2D mass_flow_rate
)
    : geom(std::move(geometry))
    , fluid(std::move(fluid))
    , T_inlet(inlet_temperature)
    , P_inlet(inlet_pressure)
{
    // initialize solution vectors
    state.fluid = *this->fluid; // set fluid reference in state

    size_t nx = geom->nx();
    size_t ny = geom->ny();
    size_t nz = geom->naxial() + 1;

    Vector::resize(state.h_l, nx, ny, nz);
    Vector::resize(state.P, nx, ny, nz);
    Vector::resize(state.W_l, nx, ny, nz);
    Vector::resize(state.W_v, nx, ny, nz);
    Vector::resize(state.alpha, nx, ny, nz);
    Vector::resize(state.X, nx, ny, nz);
    Vector::resize(state.lhr, nx, ny, geom->naxial());
    Vector::resize(state.evap, nx, ny, geom->naxial());

    for (size_t k = 0; k < nz; ++k) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                state.h_l[i][j][k] = fluid->h(T_inlet[i][j]);
                state.P[i][j][k] = P_inlet[i][j];
                state.W_l[i][j][k] = mass_flow_rate[i][j];
                state.lhr[i][j][k] = linear_heat_rate[i][j];
            }
        }
    }

    std::cout << "Solver initialized." << std::endl;
}

Vector3D Solver::get_evaporation_rates() const {
    Vector3D evap_rates = state.evap;
    for (size_t k = 0; k < geom->naxial(); ++k) {
        for (size_t j = 0; j < geom->ny(); ++j) {
            for (size_t i = 0; i < geom->nx(); ++i) {
                evap_rates[i][j][k] = state.evap[i][j][k] * geom->dz();
            }
        }
    }
    return evap_rates;
}

void Solver::solve() {
    for (size_t iter = 0; iter < 1000; ++iter) { // fixed number of iterations for now
        TH::solve_evaporation_term(state, *geom, *fluid);
        TH::solve_turbulent_mixing(state, *geom, *fluid);   // --- TODO: Issue #73 ---
        TH::solve_void_drift(state, *geom, *fluid);         // --- TODO: Issue #74 ---
        TH::solve_surface_mass_flux(state, *geom, *fluid);  // --- TODO: Issue #75 ---
        TH::solve_flow_rates(state, *geom, *fluid);
        TH::solve_enthalpy(state, *geom, *fluid);
        TH::solve_void_fraction(state, *geom, *fluid);
        TH::solve_quality(state, *geom, *fluid);
        TH::solve_pressure(state, *geom, *fluid);
    }
}
