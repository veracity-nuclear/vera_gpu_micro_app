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

    if (args.get_flag("verbose")) {
        _verbose = true;
    }

    // Extract parameters from ArgumentParser and initialize Solver
    std::string filename = args.get_positional(0);

    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group core = file.getGroup("CORE");
    HighFive::Group state_pt = file.getGroup("STATE_0001");

    // Load pin powers and compute linear heat rate for each subchannel
    // pin_powers shape: (npin, npin, nz, nassemblies)
    auto pin_powers = HDF5ToKokkosView<View4D>(state_pt.getDataSet("pin_powers"), "pin_powers"); // unitless, normalized pin powers

    // pin_powers are normalized, so multiply by nominal_lhr to get local linear power [W/m]
    auto nominal_lhr = core.getDataSet("nominal_linear_heat_rate").read<double>() * 100.0; // convert from W/cm to W/m
    auto percent_power = state_pt.getDataSet("power").read<double>() * 0.01; // convert from % to fraction
    nominal_lhr *= percent_power; // scale by core power level
    Kokkos::parallel_for("denormalize_pin_powers",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>, ExecutionSpace>({0,0,0,0},
            {pin_powers.extent(0), pin_powers.extent(1), pin_powers.extent(2), pin_powers.extent(3)}),
        KOKKOS_LAMBDA(const size_t i, const size_t j, const size_t k, const size_t a) {
            pin_powers(i, j, k, a) *= nominal_lhr;
        });
    Kokkos::fence();

    // Allocate state.lhr for spatially-varying linear heat rate
    size_t nz = state.geom->naxial() + 1;
    size_t nchan = state.geom->nchannels();
    size_t nsurf = state.geom->nsurfaces();
    Kokkos::resize(state.lhr, nchan, state.geom->naxial());

    // Compute LHR: each subchannel takes 1/4 power from each adjacent pin
    size_t npin = state.geom->npin();
    size_t core_size = state.geom->core_size();
    for (size_t aj = 0; aj < core_size; ++aj) {
        for (size_t ai = 0; ai < core_size; ++ai) {
            if (state.geom->core_map(aj, ai) == 0) continue;
            size_t assem_idx = state.geom->core_map(aj, ai) - 1;

            for (size_t j = 0; j < state.geom->nchan(); ++j) {
                for (size_t i = 0; i < state.geom->nchan(); ++i) {
                    size_t aij = state.geom->global_chan_index(aj, ai, j, i);

                    for (size_t k = 0; k < state.geom->naxial(); ++k) {
                        double pin_power_sum = 0.0;

                        // Check 4 neighboring pins (SW, SE, NW, NE)
                        // Subchannel (j,i) is bounded by pins:
                        // SW: (j, i-1), SE: (j, i), NW: (j-1, i-1), NE: (j-1, i)

                        if (j > 0 && i > 0) { // NW pin exists
                            pin_power_sum += 0.25 * pin_powers(i - 1, j - 1, k, assem_idx);
                        }
                        if (j > 0 && i < npin) { // NE pin exists
                            pin_power_sum += 0.25 * pin_powers(i, j - 1, k, assem_idx);
                        }
                        if (j < npin && i > 0) { // SW pin exists
                            pin_power_sum += 0.25 * pin_powers(i - 1, j, k, assem_idx);
                        }
                        if (j < npin && i < npin) { // SE pin exists
                            pin_power_sum += 0.25 * pin_powers(i, j, k, assem_idx);
                        }

                        // pin_power_sum is already in [W/m] after denormalization above
                        state.lhr(aij, k) = pin_power_sum;
                    }
                }
            }
        }
    }

    // extract inlet temperature
    double inlet_temp = state_pt.getDataSet("core_inlet_temp").read<double>() + 273.15; // convert to K
    View1D inlet_temperature = View1D("inlet_temperature", state.geom->nchannels());
    for (size_t ij = 0; ij < inlet_temperature.extent(0); ++ij) {
        inlet_temperature(ij) = inlet_temp;
    }

    // extract inlet pressure and mass flow rate
    auto channel_pressure = HDF5ToKokkosView<View4D>(state_pt.getDataSet("channel_pressure"), "channel_pressure");
    auto channel_pressure_inlet = Kokkos::subview(channel_pressure, Kokkos::ALL(), Kokkos::ALL(), 0, Kokkos::ALL());
    View1D inlet_pressure("inlet_pressure", state.geom->nchannels());

    // /STATE_0001/pressure [MPa]
    double pressure = state_pt.getDataSet("pressure").read<double>() * 1e6; // convert to Pa

    // REFACTOR for mass flow rate: assembly-wise mass flow rates exist in VERAout
    View1D mass_flow_rate("mass_flow_rate", state.geom->nchannels());
    // /STATE_0001/flow_dist {560}
    View1D flow_dist = HDF5ToKokkosView<View1D>(state_pt.getDataSet("flow_dist"), "flow_dist");
    // /STATE_0001/flow {SCALAR} [%] % of rated flow
    double flow_percent = state_pt.getDataSet("flow").read<double>() * 0.01;
    // /CORE/rated_flow {SCALAR} [kg/s] Rated vessel flow at 100% flow
    double rated_flow = state_pt.getDataSet("rated_flow").read<double>(); // kg/s
    double total_mass_flow = rated_flow * flow_percent;
    double avg_assy_mass_flow = total_mass_flow / state.geom->nassemblies();


    std::cout << "Total mass flow = " << total_mass_flow << " kg/s" << std::endl;
    std::cout << "Avg assembly mass flow = " << avg_assy_mass_flow << " kg/s" << std::endl;
    std::cout << "Avg channel mass flow = " << avg_assy_mass_flow / (state.geom->nchan() * state.geom->nchan()) << " kg/s" << std::endl;

    size_t assy_idx = 0;
    for (size_t aj = 0; aj < state.geom->core_size(); ++aj) {
        for (size_t ai = 0; ai < state.geom->core_size(); ++ai) {
            if (state.geom->core_map(aj, ai) == 0) continue;

            double assembly_mdot = avg_assy_mass_flow * flow_dist(assy_idx); // assembly-by-assembly mass flow rate
            double mdot = assembly_mdot / (state.geom->nchan() * state.geom->nchan());

            size_t assem = state.geom->core_map(aj, ai);
            for (size_t j = 0; j < state.geom->nchan(); ++j) {
                for (size_t i = 0; i < state.geom->nchan(); ++i) {
                    size_t aij = state.geom->global_chan_index(aj, ai, j, i);
                    inlet_pressure(aij) = pressure; // Pa
                    mass_flow_rate(aij) = mdot; // kg/s
                }
            }

            assy_idx++;
        }
    }

    // Compute average inlet pressure and total mass flow rate
    double avg_inlet_pressure = 0.0;
    double total_mass_flow_rate = 0.0;
    for (size_t i = 0; i < inlet_pressure.extent(0); ++i) {
        avg_inlet_pressure += inlet_pressure(i);
        total_mass_flow_rate += mass_flow_rate(i);
    }
    avg_inlet_pressure /= inlet_pressure.extent(0);

    std::cout << "Inlet Temperature: " << inlet_temp << " K" << std::endl;
    std::cout << "Inlet Pressure (avg): " << avg_inlet_pressure / 1e3 << " kPa" << std::endl;
    std::cout << "Mass Flow Rate (total): " << total_mass_flow_rate << " kg/s" << std::endl;

    // Create host mirror to read LHR for printing
    auto h_lhr_print = Kokkos::create_mirror_view(state.lhr);
    Kokkos::deep_copy(h_lhr_print, state.lhr);

    // initialize solution vectors
    Kokkos::resize(state.h_l, nchan, nz);
    Kokkos::resize(state.P, nchan, nz);
    Kokkos::resize(state.W_l, nchan, nz);
    Kokkos::resize(state.W_v, nchan, nz);
    Kokkos::resize(state.alpha, nchan, nz);
    Kokkos::resize(state.X, nchan, nz);
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
    auto h_mass_flow_rate = Kokkos::create_mirror_view(mass_flow_rate);

    // Copy input data to host
    Kokkos::deep_copy(h_inlet_temperature, inlet_temperature);
    Kokkos::deep_copy(h_inlet_pressure, inlet_pressure);
    Kokkos::deep_copy(h_mass_flow_rate, mass_flow_rate);

    // set inlet boundary conditions for surface quantities (0 to naxial)
    for (size_t k = 0; k < nz; ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            h_h_l(i, k) = state.fluid->h(h_inlet_temperature(i));
            h_P(i, k) = h_inlet_pressure(i);
            h_W_l(i, k) = h_mass_flow_rate(i);
        }
    }

    // Note: h_lhr is already populated from pin_powers above, no need to set here

    // Copy initialized data back to device
    Kokkos::deep_copy(state.h_l, h_h_l);
    Kokkos::deep_copy(state.P, h_P);
    Kokkos::deep_copy(state.W_l, h_W_l);
    Kokkos::deep_copy(state.lhr, h_lhr);

    // Print solver initialization statistics
    std::cout << "\n=== SOLVER INITIALIZATION SUMMARY ===" << std::endl;
    std::cout << "Execution resources: " << ExecutionSpace::concurrency() << std::endl;
    std::cout << "Verbose mode: " << (_verbose ? "ON" : "OFF") << std::endl;
    std::cout << "Crossflow: " << (_cf_flag ? "ENABLED" : "DISABLED") << std::endl;

    // Compute inlet enthalpy
    double h_in = state.fluid->h(inlet_temp);
    std::cout << "\nInlet Conditions:" << std::endl;
    std::cout << "  Temperature: " << inlet_temp - 273.15 << " °C (" << inlet_temp << " K)" << std::endl;
    std::cout << "  Enthalpy: " << h_in / 1e3 << " kJ/kg" << std::endl;

    // LHR statistics
    double lhr_min = std::numeric_limits<double>::max();
    double lhr_max = std::numeric_limits<double>::lowest();
    double lhr_sum = 0.0;
    double total_heat_generated = 0.0;
    size_t lhr_count = 0;
    for (size_t aij = 0; aij < state.geom->nchannels(); ++aij) {
        for (size_t k = 0; k < state.geom->naxial(); ++k) {
            double val = h_lhr(aij, k);
            if (val > 1e-12) {
                lhr_min = std::min(lhr_min, val);
                lhr_max = std::max(lhr_max, val);
                lhr_sum += val;
                total_heat_generated += val * state.geom->dz(k);
                lhr_count++;
            }
        }
    }
    double lhr_avg = lhr_sum / lhr_count;
    std::cout << "\nLinear Heat Rate [W/cm]:" << std::endl;
    std::cout << "  Min: " << lhr_min / 1e2 << ", Max: " << lhr_max / 1e2 << ", Avg: " << lhr_avg / 1e2 << std::endl;
    std::cout << "Total power: " << total_heat_generated / 1e6 << " MW" << std::endl;
    std::cout << "=====================================\n" << std::endl;

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

    // Print solver initialization statistics
    std::cout << "\n=== SOLVER INITIALIZATION SUMMARY ===" << std::endl;
    std::cout << "Execution resources: " << ExecutionSpace::concurrency() << std::endl;
    std::cout << "Verbose mode: " << (_verbose ? "ON" : "OFF") << std::endl;
    std::cout << "Crossflow: " << (_cf_flag ? "ENABLED" : "DISABLED") << std::endl;

    // Pressure statistics
    double P_min = std::numeric_limits<double>::max();
    double P_max = std::numeric_limits<double>::lowest();
    double P_sum = 0.0;
    for (size_t aij = 0; aij < state.geom->nchannels(); ++aij) {
        for (size_t k = 0; k <= state.geom->naxial(); ++k) {
            double val = h_P(aij, k);
            P_min = std::min(P_min, val);
            P_max = std::max(P_max, val);
            P_sum += val;
        }
    }
    double P_avg = P_sum / (state.geom->nchannels() * (state.geom->naxial() + 1));
    std::cout << "\nPressure [MPa]:" << std::endl;
    std::cout << "  Min: " << P_min / 1e6 << ", Max: " << P_max / 1e6 << ", Avg: " << P_avg / 1e6 << std::endl;

    // Mass flux statistics
    double W_min = std::numeric_limits<double>::max();
    double W_max = std::numeric_limits<double>::lowest();
    double W_sum = 0.0;
    for (size_t aij = 0; aij < state.geom->nchannels(); ++aij) {
        for (size_t k = 0; k <= state.geom->naxial(); ++k) {
            double val = h_W_l(aij, k);
            W_min = std::min(W_min, val);
            W_max = std::max(W_max, val);
            W_sum += val;
        }
    }
    double W_avg = W_sum / (state.geom->nchannels() * (state.geom->naxial() + 1));
    std::cout << "\nMass Flow Rate [kg/s]:" << std::endl;
    std::cout << "  Min: " << W_min << ", Max: " << W_max << ", Avg: " << W_avg << std::endl;

    // Enthalpy and temperature statistics
    double h_min = std::numeric_limits<double>::max();
    double h_max = std::numeric_limits<double>::lowest();
    double h_sum = 0.0;
    double T_min = std::numeric_limits<double>::max();
    double T_max = std::numeric_limits<double>::lowest();
    double T_sum = 0.0;
    for (size_t aij = 0; aij < state.geom->nchannels(); ++aij) {
        for (size_t k = 0; k <= state.geom->naxial(); ++k) {
            double h_val = h_h_l(aij, k);
            h_min = std::min(h_min, h_val);
            h_max = std::max(h_max, h_val);
            h_sum += h_val;
            double T_val = state.fluid->T(h_val);
            T_min = std::min(T_min, T_val);
            T_max = std::max(T_max, T_val);
            T_sum += T_val;
        }
    }
    double h_avg = h_sum / (state.geom->nchannels() * (state.geom->naxial() + 1));
    double T_avg = T_sum / (state.geom->nchannels() * (state.geom->naxial() + 1));
    std::cout << "\nEnthalpy [kJ/kg]:" << std::endl;
    std::cout << "  Min: " << h_min / 1e3 << ", Max: " << h_max / 1e3 << ", Avg: " << h_avg / 1e3 << std::endl;
    std::cout << "\nTemperature [K]:" << std::endl;
    std::cout << "  Min: " << T_min << ", Max: " << T_max << ", Avg: " << T_avg << std::endl;

    // LHR statistics
    double lhr_min = std::numeric_limits<double>::max();
    double lhr_max = std::numeric_limits<double>::lowest();
    double lhr_sum = 0.0;
    size_t lhr_count = 0;
    for (size_t aij = 0; aij < state.geom->nchannels(); ++aij) {
        for (size_t k = 0; k < state.geom->naxial(); ++k) {
            double val = h_lhr(aij, k);
            if (val > 1e-12) {
                lhr_min = std::min(lhr_min, val);
                lhr_max = std::max(lhr_max, val);
                lhr_sum += val;
                lhr_count++;
            }
        }
    }
    double lhr_avg = lhr_sum / lhr_count;
    std::cout << "\nLinear Heat Rate [kW/m]:" << std::endl;
    std::cout << "  Min: " << lhr_min / 1e3 << ", Max: " << lhr_max / 1e3 << ", Avg: " << lhr_avg / 1e3 << std::endl;
    std::cout << "  Total power: " << lhr_sum * state.geom->core_height() / 1e6 << " MW" << std::endl;
    std::cout << "=====================================\n" << std::endl;
}

template <typename ExecutionSpace>
typename Solver<ExecutionSpace>::View2D Solver<ExecutionSpace>::get_evaporation_rates() const {
    View2D evap_rates("evap_rates", state.evap.extent(0), state.evap.extent(1));
    auto h_evap_rates = Kokkos::create_mirror_view(evap_rates);
    auto h_evap = Kokkos::create_mirror_view(state.evap);
    Kokkos::deep_copy(h_evap, state.evap);

    for (size_t k = 0; k < state.geom->naxial(); ++k) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            h_evap_rates(i, k) = h_evap(i, k) * state.geom->dz(k); // variable axial spacing
        }
    }

    Kokkos::deep_copy(evap_rates, h_evap_rates);
    return evap_rates;
}

template <typename ExecutionSpace>
void Solver<ExecutionSpace>::solve(size_t max_outer_iter, size_t max_inner_iter) {

    state.surface_plane = 0; // start at inlet axial plane
    state.node_plane = 0;    // start at first node axial plane
    state.max_outer_iter = max_outer_iter;
    state.max_inner_iter = max_inner_iter;

    print_state_at_plane(0);

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

        if (_verbose) {
            print_state_at_plane(k);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Completed axial plane " << std::setw(3) << k << "  / " <<  std::setw(3) << state.geom->naxial()
                  << std::setw(8) << duration.count() * 1e-3 << " s" << std::endl;

        // if (k >= 1) return;

    }
}

template <typename ExecutionSpace>
void Solver<ExecutionSpace>::print_state_at_plane(size_t k) {

    // Create host mirrors to access data
    auto h_h_l = Kokkos::create_mirror_view(state.h_l);
    auto h_P = Kokkos::create_mirror_view(state.P);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_alpha = Kokkos::create_mirror_view(state.alpha);
    auto h_X = Kokkos::create_mirror_view(state.X);
    auto h_lhr = Kokkos::create_mirror_view(state.lhr);
    auto h_evap = Kokkos::create_mirror_view(state.evap);

    Kokkos::deep_copy(h_h_l, state.h_l);
    Kokkos::deep_copy(h_P, state.P);
    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_alpha, state.alpha);
    Kokkos::deep_copy(h_X, state.X);
    Kokkos::deep_copy(h_lhr, state.lhr);
    Kokkos::deep_copy(h_evap, state.evap);

    std::cout << "\n=== PLANE " << k << " (Surface) ===" << std::endl;
    std::cout << std::setw(25) << "Variable" << std::setw(15) << "Min" << std::setw(15) << "Max" << std::setw(15) << "Avg" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    // Compute statistics for Liquid Enthalpy
    double h_l_min = std::numeric_limits<double>::max();
    double h_l_max = std::numeric_limits<double>::lowest();
    double h_l_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double val = h_h_l(i, k);
        h_l_min = std::min(h_l_min, val);
        h_l_max = std::max(h_l_max, val);
        h_l_sum += val;
    }
    double h_l_avg = h_l_sum / state.geom->nchannels();
    std::cout << std::setw(25) << "Enthalpy [kJ/kg]" << std::setw(15) << std::setprecision(6) << h_l_min / 1e3
              << std::setw(15) << h_l_max / 1e3 << std::setw(15) << h_l_avg / 1e3 << std::endl;

    // Compute statistics for Temperature (derived from enthalpy)
    double T_min = std::numeric_limits<double>::max();
    double T_max = std::numeric_limits<double>::lowest();
    double T_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double T_val = state.fluid->T(h_h_l(i, k));
        T_min = std::min(T_min, T_val);
        T_max = std::max(T_max, T_val);
        T_sum += T_val;
    }
    double T_avg = T_sum / state.geom->nchannels();
    std::cout << std::setw(25) << "Temperature [K]" << std::setw(15) << T_min
              << std::setw(15) << T_max << std::setw(15) << T_avg << std::endl;

    // Compute statistics for Pressure
    double P_min = std::numeric_limits<double>::max();
    double P_max = std::numeric_limits<double>::lowest();
    double P_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double val = h_P(i, k);
        P_min = std::min(P_min, val);
        P_max = std::max(P_max, val);
        P_sum += val;
    }
    double P_avg = P_sum / state.geom->nchannels();
    std::cout << std::setw(25) << "Pressure [MPa]" << std::setw(15) << std::setprecision(6) << P_min / 1e6
              << std::setw(15) << P_max / 1e6 << std::setw(15) << P_avg / 1e6 << std::endl;

    // Compute statistics for Liquid Flow Rate
    double W_l_min = std::numeric_limits<double>::max();
    double W_l_max = std::numeric_limits<double>::lowest();
    double W_l_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double val = h_W_l(i, k);
        W_l_min = std::min(W_l_min, val);
        W_l_max = std::max(W_l_max, val);
        W_l_sum += val;
    }
    double W_l_avg = W_l_sum / state.geom->nchannels();
    std::cout << std::setw(25) << "W_l [kg/s]" << std::setw(15) << W_l_min
              << std::setw(15) << W_l_max << std::setw(15) << W_l_avg << std::endl;

    // Compute statistics for Vapor Flow Rate
    double W_v_min = std::numeric_limits<double>::max();
    double W_v_max = std::numeric_limits<double>::lowest();
    double W_v_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double val = h_W_v(i, k);
        W_v_min = std::min(W_v_min, val);
        W_v_max = std::max(W_v_max, val);
        W_v_sum += val;
    }
    double W_v_avg = W_v_sum / state.geom->nchannels();
    std::cout << std::setw(25) << "W_v [kg/s]" << std::setw(15) << W_v_min
              << std::setw(15) << W_v_max << std::setw(15) << W_v_avg << std::endl;

    // Compute statistics for Mass Flux (liquid + vapor)
    double G_min = std::numeric_limits<double>::max();
    double G_max = std::numeric_limits<double>::lowest();
    double G_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double A_f = state.geom->flow_area(i, k);
        if (A_f > 1e-12) {
            double val = (h_W_l(i, k) + h_W_v(i, k)) / A_f;
            G_min = std::min(G_min, val);
            G_max = std::max(G_max, val);
            G_sum += val;
        }
    }
    double G_avg = G_sum / state.geom->nchannels();
    std::cout << std::setw(26) << "Mass Flux [kg/m²/s]" << std::setw(15) << G_min
              << std::setw(15) << G_max << std::setw(15) << G_avg << std::endl;

    // Compute statistics for Void Fraction
    double alpha_min = std::numeric_limits<double>::max();
    double alpha_max = std::numeric_limits<double>::lowest();
    double alpha_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double val = h_alpha(i, k);
        alpha_min = std::min(alpha_min, val);
        alpha_max = std::max(alpha_max, val);
        alpha_sum += val;
    }
    double alpha_avg = alpha_sum / state.geom->nchannels();
    std::cout << std::setw(25) << "Void Fraction [-]" << std::setw(15) << alpha_min
              << std::setw(15) << alpha_max << std::setw(15) << alpha_avg << std::endl;

    // Compute statistics for Quality
    double X_min = std::numeric_limits<double>::max();
    double X_max = std::numeric_limits<double>::lowest();
    double X_sum = 0.0;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double val = h_X(i, k);
        X_min = std::min(X_min, val);
        X_max = std::max(X_max, val);
        X_sum += val;
    }
    double X_avg = X_sum / state.geom->nchannels();
    std::cout << std::setw(25) << "Quality [-]" << std::setw(15) << X_min
              << std::setw(15) << X_max << std::setw(15) << X_avg << std::endl;

    std::cout << std::string(70, '-') << std::endl;

    // Print node-centered variables (k-1 since node_plane lags surface_plane)
    if (k > 0) {
        size_t k_node = k - 1;
        std::cout << "\n=== NODE " << k_node << " ===" << std::endl;
        std::cout << std::setw(25) << "Variable" << std::setw(15) << "Min" << std::setw(15) << "Max" << std::setw(15) << "Avg" << std::endl;
        std::cout << std::string(70, '-') << std::endl;

        // Compute statistics for Linear Heat Rate
        double lhr_min = std::numeric_limits<double>::max();
        double lhr_max = std::numeric_limits<double>::lowest();
        double lhr_sum = 0.0;
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            double val = h_lhr(i, k_node);
            lhr_min = std::min(lhr_min, val);
            lhr_max = std::max(lhr_max, val);
            lhr_sum += val;
        }
        double lhr_avg = lhr_sum / state.geom->nchannels();
        std::cout << std::setw(25) << "LHR [W/cm]" << std::setw(15) << lhr_min / 1e2
                  << std::setw(15) << lhr_max / 1e2 << std::setw(15) << lhr_avg / 1e2 << std::endl;

        // Compute statistics for Evaporation Rate
        double evap_min = std::numeric_limits<double>::max();
        double evap_max = std::numeric_limits<double>::lowest();
        double evap_sum = 0.0;
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            double val = h_evap(i, k_node);
            evap_min = std::min(evap_min, val);
            evap_max = std::max(evap_max, val);
            evap_sum += val;
        }
        double evap_avg = evap_sum / state.geom->nchannels();
        std::cout << std::setw(25) << "Evap [kg/m/s]" << std::setw(15) << evap_min
                  << std::setw(15) << evap_max << std::setw(15) << evap_avg << std::endl;

        std::cout << std::string(70, '-') << std::endl;
    }

}

// Explicit template instantiations
template class Solver<Kokkos::DefaultExecutionSpace>;
template class Solver<Kokkos::Serial>;
#if defined(KOKKOS_ENABLE_SERIAL) && !defined(KOKKOS_ENABLE_OPENMP)
template class Solver<Kokkos::Serial>;
#endif
