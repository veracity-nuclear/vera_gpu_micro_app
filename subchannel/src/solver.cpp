#include "solver.hpp"

Solver::Solver(
    std::shared_ptr<Geometry> geometry,
    std::shared_ptr<Water> fluid,
    Vector2D inlet_temperature,
    Vector2D inlet_pressure,
    Vector2D linear_heat_rate,
    Vector2D mass_flow_rate
)
    : T_inlet(inlet_temperature)
    , P_inlet(inlet_pressure)
{
    state.geom = geometry;
    state.fluid = fluid;

    size_t nx = state.geom->nx();
    size_t ny = state.geom->ny();
    size_t nz = state.geom->naxial() + 1;
    size_t ns = 4; // number of neighboring surfaces on an axial plane
    size_t nsurf = state.geom->nsurfaces();

    // initialize solution vectors
    Vector::resize(state.h_l, nx, ny, nz);
    Vector::resize(state.P, nx, ny, nz);
    Vector::resize(state.W_l, nx, ny, nz);
    Vector::resize(state.W_v, nx, ny, nz);
    Vector::resize(state.alpha, nx, ny, nz);
    Vector::resize(state.X, nx, ny, nz);
    Vector::resize(state.lhr, nx, ny, state.geom->naxial());
    Vector::resize(state.evap, nx, ny, state.geom->naxial());

    // initialize surface source term vectors
    Vector::resize(state.G_l_tm, nx, ny, nz, ns);
    Vector::resize(state.G_v_tm, nx, ny, nz, ns);
    Vector::resize(state.Q_m_tm, nx, ny, nz, ns);
    Vector::resize(state.M_m_tm, nx, ny, nz, ns);
    Vector::resize(state.G_l_vd, nx, ny, nz, ns);
    Vector::resize(state.G_v_vd, nx, ny, nz, ns);
    Vector::resize(state.Q_m_vd, nx, ny, nz, ns);
    Vector::resize(state.M_m_vd, nx, ny, nz, ns);

    // set inlet boundary conditions
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
    for (size_t k = 0; k < state.geom->naxial(); ++k) {
        for (size_t j = 0; j < state.geom->ny(); ++j) {
            for (size_t i = 0; i < state.geom->nx(); ++i) {
                evap_rates[i][j][k] = state.evap[i][j][k] * state.geom->dz();
            }
        }
    }
    return evap_rates;
}

void Solver::solve(size_t max_outer_iter, size_t max_inner_iter) {

    state.surface_plane = 0; // start at inlet axial plane

    // loop over axial planes
    for (size_t k = 1; k < state.geom->naxial() + 1; ++k) {

        // set current axial planes in state
        state.node_plane = k - 1;

        // closure relations
        TH::solve_evaporation_term(state);
        // TH::solve_turbulent_mixing(state);
        // TH::solve_void_drift(state);

        // closure relations use lagging edge values, so update after solving them
        state.surface_plane = k;

        // outer iteration (solution for full axial plane)
        for (size_t outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {

            TH::solve_surface_mass_flux(state);

            // inner iteration
            for (size_t inner_iter = 0; inner_iter < max_inner_iter; ++inner_iter) {
                TH::solve_flow_rates(state);
                TH::solve_enthalpy(state);
                TH::solve_void_fraction(state);
                TH::solve_quality(state);
                TH::solve_pressure(state);
            }
        }
    }
}
