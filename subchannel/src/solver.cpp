#include "solver.hpp"
#include <chrono>

template <typename ExecutionSpace>
Solver<ExecutionSpace>::Solver(const ArgumentParser& args) {

    state.geom = std::make_shared<Geometry<ExecutionSpace>>(args);
    state.fluid = std::make_shared<Water<ExecutionSpace>>();

    // check for crossflow flag
    if (args.get_flag("no-crossflow")) {
        _cf_flag = false;
    }

    // Extract parameters from ArgumentParser and initialize Solver
    std::string filename = args.get_positional(0);

    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group core = file.getGroup("CORE");
    HighFive::Group state_pt = file.getGroup("STATE_0001");

    // extract inlet temperature
    double inlet_temp = state_pt.getDataSet("core_inlet_temp").read<double>() + 273.15; // convert to K
    View1D inlet_temperature = View1D("inlet_temperature", state.geom->nchannels());
    for (size_t ij = 0; ij < inlet_temperature.extent(0); ++ij) {
        inlet_temperature(ij) = inlet_temp;
    }

    // extract linear heat rate for each pin
    double nominal_lhr = core.getDataSet("nominal_linear_heat_rate").read<double>(); // W/cm, placeholder value
    View1D linear_heat_rate = View1D("linear_heat_rate", state.geom->nchannels());
    for (size_t ij = 0; ij < linear_heat_rate.extent(0); ++ij) {
        linear_heat_rate(ij) = nominal_lhr;
    }

    // extract inlet pressure and mass flow rate
    auto channel_pressure = HDF5ToKokkosView<View4D>(state_pt.getDataSet("channel_pressure"), "channel_pressure");
    auto channel_pressure_inlet = Kokkos::subview(channel_pressure, Kokkos::ALL(), Kokkos::ALL(), 0, Kokkos::ALL());
    View1D inlet_pressure("inlet_pressure", state.geom->nchannels());

    auto Af = HDF5ToKokkosView<View4D>(core.getDataSet("channel_area"), "channel_area"); // cm2
    auto Af_inlet = Kokkos::subview(Af, Kokkos::ALL(), Kokkos::ALL(), 0, Kokkos::ALL());
    auto Gm = HDF5ToKokkosView<View4D>(state_pt.getDataSet("channel_mixture_mass_flux"), "Gm"); // kg/m2/s
    auto Gm_inlet = Kokkos::subview(Gm, Kokkos::ALL(), Kokkos::ALL(), 0, Kokkos::ALL());
    View1D mass_flow_rate("mass_flow_rate", state.geom->nchannels());

    size_t idx = 0;
    for (size_t aj = 0; aj < state.geom->core_size(); ++aj) {
        for (size_t ai = 0; ai < state.geom->core_size(); ++ai) {
            if (state.geom->core_map(aj, ai) == 0) continue;

            size_t assem = state.geom->core_map(aj, ai);
            for (size_t j = 0; j < state.geom->nchan(); ++j) {
                for (size_t i = 0; i < state.geom->nchan(); ++i) {
                    inlet_pressure(idx) = channel_pressure_inlet(i, j, assem) * 1e5; // convert bar to Pa
                    size_t mid_chan = state.geom->nchan() / 2;
                    mass_flow_rate(idx) = Af_inlet(mid_chan, mid_chan, 0) * Gm_inlet(mid_chan, mid_chan, 0) * 1e-4; // convert kg/m2/s to kg/cm2/s
                    ++idx;
                }
            }
        }
    }

    std::cout << "Inlet Temperature: " << inlet_temp << " K" << std::endl;
    std::cout << "Inlet Pressure: " << inlet_pressure(0) << " Pa" << std::endl;
    std::cout << "Mass Flow Rate: " << mass_flow_rate(0) << " kg/s" << std::endl;
    std::cout << "Linear Heat Rate: " << linear_heat_rate(0) << " W/cm" << std::endl;

    size_t nz = state.geom->naxial() + 1;
    size_t nchan = state.geom->nchannels();
    size_t nsurf = state.geom->nsurfaces();

    // initialize solution vectors
    Kokkos::resize(state.h_l, nchan, nz);
    Kokkos::resize(state.P, nchan, nz);
    Kokkos::resize(state.W_l, nchan, nz);
    Kokkos::resize(state.W_v, nchan, nz);
    Kokkos::resize(state.alpha, nchan, nz);
    Kokkos::resize(state.X, nchan, nz);
    Kokkos::resize(state.lhr, nchan, state.geom->naxial());
    Kokkos::resize(state.evap, nchan, state.geom->naxial());

    // initialize surface source term vectors
    Kokkos::resize(state.G_l_tm, nsurf);
    Kokkos::resize(state.G_v_tm, nsurf);
    Kokkos::resize(state.Q_m_tm, nsurf);
    Kokkos::resize(state.M_m_tm, nsurf);
    Kokkos::resize(state.G_l_vd, nsurf);
    Kokkos::resize(state.G_v_vd, nsurf);
    Kokkos::resize(state.Q_m_vd, nsurf);
    Kokkos::resize(state.M_m_vd, nsurf);
    Kokkos::resize(state.gk, nsurf, state.geom->naxial());

    // Create host mirrors for initialization
    auto h_h_l = Kokkos::create_mirror_view(state.h_l);
    auto h_P = Kokkos::create_mirror_view(state.P);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_lhr = Kokkos::create_mirror_view(state.lhr);
    auto h_inlet_temperature = Kokkos::create_mirror_view(inlet_temperature);
    auto h_inlet_pressure = Kokkos::create_mirror_view(inlet_pressure);
    auto h_linear_heat_rate = Kokkos::create_mirror_view(linear_heat_rate);
    auto h_mass_flow_rate = Kokkos::create_mirror_view(mass_flow_rate);

    // Copy input data to host
    Kokkos::deep_copy(h_inlet_temperature, inlet_temperature);
    Kokkos::deep_copy(h_inlet_pressure, inlet_pressure);
    Kokkos::deep_copy(h_linear_heat_rate, linear_heat_rate);
    Kokkos::deep_copy(h_mass_flow_rate, mass_flow_rate);

    // set inlet boundary conditions for surface quantities (0 to naxial)
    for (size_t k = 0; k < nz; ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            h_h_l(i, k) = state.fluid->h(h_inlet_temperature(i));
            h_P(i, k) = h_inlet_pressure(i);
            h_W_l(i, k) = h_mass_flow_rate(i);
        }
    }

    // set node quantities (0 to naxial-1)
    for (size_t k = 0; k < state.geom->naxial(); ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            h_lhr(i, k) = h_linear_heat_rate(i);
        }
    }

    // Copy initialized data back to device
    Kokkos::deep_copy(state.h_l, h_h_l);
    Kokkos::deep_copy(state.P, h_P);
    Kokkos::deep_copy(state.W_l, h_W_l);
    Kokkos::deep_copy(state.lhr, h_lhr);

    std::cout << "Solver initialized (using " << ExecutionSpace::concurrency() << " execution resources)." << std::endl;

}

template <typename ExecutionSpace>
Solver<ExecutionSpace>::Solver(
    std::shared_ptr<Geometry<ExecutionSpace>> geometry,
    std::shared_ptr<Water<ExecutionSpace>> fluid,
    View1D inlet_temperature,
    View1D inlet_pressure,
    View1D linear_heat_rate,
    View1D mass_flow_rate
) {
    state.geom = geometry;
    state.fluid = fluid;

    size_t nz = state.geom->naxial() + 1;
    size_t nchan = state.geom->nchannels();
    size_t nsurf = state.geom->nsurfaces();

    // initialize solution vectors
    Kokkos::resize(state.h_l, nchan, nz);
    Kokkos::resize(state.P, nchan, nz);
    Kokkos::resize(state.W_l, nchan, nz);
    Kokkos::resize(state.W_v, nchan, nz);
    Kokkos::resize(state.alpha, nchan, nz);
    Kokkos::resize(state.X, nchan, nz);
    Kokkos::resize(state.lhr, nchan, state.geom->naxial());
    Kokkos::resize(state.evap, nchan, state.geom->naxial());

    // initialize surface source term vectors
    Kokkos::resize(state.G_l_tm, nsurf);
    Kokkos::resize(state.G_v_tm, nsurf);
    Kokkos::resize(state.Q_m_tm, nsurf);
    Kokkos::resize(state.M_m_tm, nsurf);
    Kokkos::resize(state.G_l_vd, nsurf);
    Kokkos::resize(state.G_v_vd, nsurf);
    Kokkos::resize(state.Q_m_vd, nsurf);
    Kokkos::resize(state.M_m_vd, nsurf);
    Kokkos::resize(state.gk, nsurf, state.geom->naxial());

    // Create host mirrors for initialization
    auto h_h_l = Kokkos::create_mirror_view(state.h_l);
    auto h_P = Kokkos::create_mirror_view(state.P);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_lhr = Kokkos::create_mirror_view(state.lhr);
    auto h_inlet_temperature = Kokkos::create_mirror_view(inlet_temperature);
    auto h_inlet_pressure = Kokkos::create_mirror_view(inlet_pressure);
    auto h_linear_heat_rate = Kokkos::create_mirror_view(linear_heat_rate);
    auto h_mass_flow_rate = Kokkos::create_mirror_view(mass_flow_rate);

    // Copy input data to host
    Kokkos::deep_copy(h_inlet_temperature, inlet_temperature);
    Kokkos::deep_copy(h_inlet_pressure, inlet_pressure);
    Kokkos::deep_copy(h_linear_heat_rate, linear_heat_rate);
    Kokkos::deep_copy(h_mass_flow_rate, mass_flow_rate);

    // set inlet boundary conditions for surface quantities (0 to naxial)
    for (size_t k = 0; k < nz; ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            h_h_l(i, k) = fluid->h(h_inlet_temperature(i));
            h_P(i, k) = h_inlet_pressure(i);
            h_W_l(i, k) = h_mass_flow_rate(i);
        }
    }

    // set node quantities (0 to naxial-1)
    for (size_t k = 0; k < state.geom->naxial(); ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            h_lhr(i, k) = h_linear_heat_rate(i);
        }
    }

    // Copy initialized data back to device
    Kokkos::deep_copy(state.h_l, h_h_l);
    Kokkos::deep_copy(state.P, h_P);
    Kokkos::deep_copy(state.W_l, h_W_l);
    Kokkos::deep_copy(state.lhr, h_lhr);

    std::cout << "Solver initialized (using " << ExecutionSpace::concurrency() << " execution resources)." << std::endl;
}

template <typename ExecutionSpace>
typename Solver<ExecutionSpace>::View2D Solver<ExecutionSpace>::get_evaporation_rates() const {
    View2D evap_rates("evap_rates", state.evap.extent(0), state.evap.extent(1));
    auto h_evap_rates = Kokkos::create_mirror_view(evap_rates);
    auto h_evap = Kokkos::create_mirror_view(state.evap);
    Kokkos::deep_copy(h_evap, state.evap);

    for (size_t k = 0; k < state.geom->naxial(); ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            h_evap_rates(i, k) = h_evap(i, k) * state.geom->dz();
        }
    }

    Kokkos::deep_copy(evap_rates, h_evap_rates);
    return evap_rates;
}

template <typename ExecutionSpace>
void Solver<ExecutionSpace>::solve(size_t max_outer_iter, size_t max_inner_iter, bool debug) {

    state.surface_plane = 0; // start at inlet axial plane
    state.node_plane = 0;    // start at first node axial plane
    state.max_outer_iter = max_outer_iter;
    state.max_inner_iter = max_inner_iter;

    // loop over axial planes
    for (size_t k = 1; k < state.geom->naxial() + 1; ++k) {

        auto start_time = std::chrono::high_resolution_clock::now();

        // set current axial planes in state
        state.node_plane = k - 1;

        // closure relations
        TH::solve_evaporation_term<ExecutionSpace>(state);
        TH::solve_mixing<ExecutionSpace>(state);

        // closure relations use lagging edge values, so update after solving them
        state.surface_plane = k;

        if (_cf_flag) {
            TH::solve_surface_mass_flux<ExecutionSpace>(state);
        } else {
            TH::planar<ExecutionSpace>(state);
        }

        if (debug) {
            print_state_at_plane(k);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Completed axial plane " << std::setw(3) << k << "  / " <<  std::setw(3) << state.geom->naxial()
                  << std::setw(8) << duration.count() * 1e-3 << " s" << std::endl;

    }
}

template <typename ExecutionSpace>
void Solver<ExecutionSpace>::print_state_at_plane(size_t k) {

    // Create host mirrors to access data
    auto h_P = Kokkos::create_mirror_view(state.P);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_alpha = Kokkos::create_mirror_view(state.alpha);
    auto h_X = Kokkos::create_mirror_view(state.X);

    Kokkos::deep_copy(h_P, state.P);
    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_alpha, state.alpha);
    Kokkos::deep_copy(h_X, state.X);

    std::cout << "\nPLANE " << k << std::endl;

    std::cout << "\nPressure:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << h_P(i, k) / 1e6 << " ";
        if ((i + 1) % state.geom->nchan() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nLiquid Flow Rate:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << h_W_l(i, k) << " ";
        if ((i + 1) % state.geom->nchan() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nVapor Flow Rate:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << h_W_v(i, k) << " ";
        if ((i + 1) % state.geom->nchan() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nAlpha:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << h_alpha(i, k) << " ";
        if ((i + 1) % state.geom->nchan() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "\nQuality:" << std::endl;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        std::cout << std::setw(14) << h_X(i, k) << " ";
        if ((i + 1) % state.geom->nchan() == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

}

// Explicit template instantiations
template class Solver<Kokkos::DefaultExecutionSpace>;
template class Solver<Kokkos::Serial>;
#if defined(KOKKOS_ENABLE_SERIAL) && !defined(KOKKOS_ENABLE_OPENMP)
template class Solver<Kokkos::Serial>;
#endif
