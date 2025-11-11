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

void Solver::solve(size_t max_outer_iter, size_t max_inner_iter, bool debug) {

    state.surface_plane = 0; // start at inlet axial plane
    state.node_plane = 0;    // start at first node axial plane
    state.max_outer_iter = max_outer_iter;
    state.max_inner_iter = max_inner_iter;

    // loop over axial planes
    for (size_t k = 1; k < state.geom->naxial() + 1; ++k) {

        // set current axial planes in state
        state.node_plane = k - 1;

        // closure relations
        TH::solve_evaporation_term(state);
        TH::solve_mixing(state);

        // closure relations use lagging edge values, so update after solving them
        state.surface_plane = k;

        TH::solve_surface_mass_flux(state);

        if (debug) {
            print_state_at_plane(k);
        }

    }
}

void Solver::print_state_at_plane(size_t k) {

    std::cout << "\nPLANE " << k << std::endl;

    std::cout << "\nPressure:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << state.P[i][k] / 1e6 << " ";
        if ((i + 1) % state.geom->nx() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nLiquid Flow Rate:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << state.W_l[i][k] << " ";
        if ((i + 1) % state.geom->nx() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nVapor Flow Rate:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << state.W_v[i][k] << " ";
        if ((i + 1) % state.geom->nx() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nAlpha:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << state.alpha[i][k] << " ";
        if ((i + 1) % state.geom->nx() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nQuality:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << state.X[i][k] << " ";
        if ((i + 1) % state.geom->nx() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

}
