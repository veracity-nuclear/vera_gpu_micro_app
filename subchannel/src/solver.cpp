#include "solver.hpp"

Solver::Solver(
    std::shared_ptr<Geometry> geometry,
    std::shared_ptr<Water> fluid,
    Vector1D inlet_temperature,
    Vector1D inlet_pressure,
    Vector1D linear_heat_rate,
    Vector1D mass_flow_rate
) {
    state.geom = geometry;
    state.fluid = fluid;

    size_t nx = state.geom->nx();
    size_t ny = state.geom->ny();
    size_t nz = state.geom->naxial() + 1;
    size_t nchan = state.geom->nchannels();
    size_t nsurf = state.geom->nsurfaces();

    // initialize solution vectors
    Vector::resize(state.h_l, nchan, nz);
    Vector::resize(state.P, nchan, nz);
    Vector::resize(state.W_l, nchan, nz);
    Vector::resize(state.W_v, nchan, nz);
    Vector::resize(state.alpha, nchan, nz);
    Vector::resize(state.X, nchan, nz);
    Vector::resize(state.lhr, nchan, state.geom->naxial());
    Vector::resize(state.evap, nchan, state.geom->naxial());
    Vector::resize(state.qz, nchan, state.geom->naxial());

    // initialize surface source term vectors
    Vector::resize(state.G_l_tm, nsurf);
    Vector::resize(state.G_v_tm, nsurf);
    Vector::resize(state.Q_m_tm, nsurf);
    Vector::resize(state.M_m_tm, nsurf);
    Vector::resize(state.G_l_vd, nsurf);
    Vector::resize(state.G_v_vd, nsurf);
    Vector::resize(state.Q_m_vd, nsurf);
    Vector::resize(state.M_m_vd, nsurf);
    Vector::resize(state.gk, nsurf, state.geom->naxial());

    // set inlet boundary conditions for surface quantities (0 to naxial)
    for (size_t k = 0; k < nz; ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            state.h_l[i][k] = fluid->h(inlet_temperature[i]);
            state.P[i][k] = inlet_pressure[i];
            state.W_l[i][k] = mass_flow_rate[i];
        }
    }

    // set node quantities (0 to naxial-1)
    for (size_t k = 0; k < state.geom->naxial(); ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            state.lhr[i][k] = linear_heat_rate[i];
        }
    }

    std::cout << "Solver initialized." << std::endl;
}

Vector2D Solver::get_evaporation_rates() const {
    Vector2D evap_rates = state.evap;
    for (size_t k = 0; k < state.geom->naxial(); ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            evap_rates[i][k] = state.evap[i][k] * state.geom->dz();
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
        TH::solve_mixing(state);

        // closure relations use lagging edge values, so update after solving them
        state.surface_plane = k;

        // outer iteration (solution for full axial plane)
        for (size_t outer_iter = 0; outer_iter < max_outer_iter; ++outer_iter) {

            // TH::solve_surface_mass_flux(state);

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
