#include "ThermalHydraulics.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

namespace ants {
namespace subchannel {

// Global counter for Newton iterations
static int global_newton_iteration_count = 0;

ThermalHydraulics::ThermalHydraulics(std::shared_ptr<SubchannelData> data,
                                   std::shared_ptr<Properties> properties)
    : data_(data)
    , properties_(properties)
    , viscosity_liquid_(0.0001)          // [Pa-s]
    , thermal_conductivity_(0.6)         // [W/m-K]
    , prandtl_number_(1.0)
    , reynolds_number_(10000.0)
    , peclet_number_(10000.0)
{
    // Initialize APEX-style Chisholm parameters
    double rhof = 750.0;    // kg/m³
    double rhog = 40.0;     // kg/m³
    double viscf = viscosity_liquid_;  // Pa-s
    double viscg = 1.2e-5;  // Pa-s - typical vapor viscosity

    // APEX Chisholm correlation constants
    double gamasq = (rhof/rhog) * std::pow(viscg/viscf, 0.2);
    double bgamm0 = std::sqrt(gamasq);
    gamma_squared_ = gamasq - 1.0;

    if (bgamm0 <= 9.5) {
        chisholm_parameter_b_ = 0.055;
    } else if (bgamm0 >= 28.0) {
        chisholm_parameter_b_ = 15.0 / gamasq;
    } else {
        chisholm_parameter_b_ = 0.52 / bgamm0;
    }

    // Initialize working arrays
    int n_surfaces = data_->getNumSurfaces();
    int n_axial = data_->getNumAxialNodes();

    if (n_surfaces > 0) {
        old_crossflow_.resize(n_surfaces * (n_axial + 1), 0.0);
        residual_.resize(n_surfaces * (n_axial + 1), 0.0);
    }
}

void ThermalHydraulics::solve() {
    int n_channels = data_->getNumSubchannels();

    std::cout << "Starting ANTS subchannel solution..." << std::endl;
    std::cout << "Subchannels: " << n_channels << std::endl;
    std::cout << "Axial nodes: " << data_->getNumAxialNodes() << std::endl;
    std::cout << "Surfaces: " << data_->getNumSurfaces() << std::endl;

    // Reset iteration counters
    resetNewtonIterationCount();

    if (n_channels == 1) {
        std::cout << "Single channel mode - no outer iterations required" << std::endl;
        solveSingleChannel(0);
    } else {
        std::cout << "Multi-channel mode - outer iterations for crossflow convergence" << std::endl;
        solveMultiChannel();
    }

    std::cout << "Solution completed." << std::endl;
}

void ThermalHydraulics::solveSingleChannel(int channel_id) {
    // For single channel, just perform axial march
    solveAxialMarch(channel_id);
}

void ThermalHydraulics::solveMultiChannel() {
    // Outer iteration loop for crossflow convergence
    int outer_iter = 0;
    bool converged = false;

    std::cout << "Starting multi-channel solution with outer iterations..." << std::endl;

    while (!converged && outer_iter < data_->max_outer_iterations) {
        // Store old crossflow values
        if (!old_crossflow_.empty()) {
            std::copy(data_->crossflow_mass_flux.begin(),
                     data_->crossflow_mass_flux.end(),
                     old_crossflow_.begin());
        }

        // Solve axial march for all channels
        for (int ch = 0; ch < data_->getNumSubchannels(); ++ch) {
            solveAxialMarch(ch);
        }

        // Update crossflow based on pressure differences
        solveCrossflowIteration();

        // Check convergence
        converged = checkConvergence();

        // Print iteration progress
        if (!converged) {
            double max_residual = 0.0;
            if (!old_crossflow_.empty() && !data_->crossflow_mass_flux.empty()) {
                for (size_t i = 0; i < old_crossflow_.size(); ++i) {
                    double residual = std::abs(data_->crossflow_mass_flux[i] - old_crossflow_[i]);
                    max_residual = std::max(max_residual, residual);
                }
            }
            std::cout << "  Outer iteration " << outer_iter + 1 << ": Max crossflow residual = "
                      << max_residual << " (tolerance: " << data_->outer_tolerance << ")" << std::endl;
        }

        ++outer_iter;
    }

    if (converged) {
        std::cout << "Outer iterations converged in " << outer_iter << " iterations" << std::endl;
    } else {
        std::cout << "Warning: Outer iteration did not converge after "
                  << outer_iter << " iterations" << std::endl;
    }
}

void ThermalHydraulics::solveAxialMarch(int channel_id) {
    int n_axial = data_->getNumAxialNodes();

    // Track Newton iterations for statistics
    int total_newton_iterations = 0;
    int max_newton_iterations = 0;

    if (channel_id == 0) {
        std::cout << "Performing axial march for channel " << channel_id << "..." << std::endl;
    } else if (channel_id == 4) {
        std::cout << "Performing axial march for channel " << channel_id << " (CENTER - NO HEATING)..." << std::endl;
    }

    // Set inlet conditions at axial node 0
    int inlet_idx = getAxialIndex(channel_id, 0);
    data_->liquid_mass_flow[inlet_idx] = data_->mass_flow_rate;
    data_->vapor_mass_flow[inlet_idx] = 0.0;
    data_->total_mass_flow[inlet_idx] = data_->mass_flow_rate;
    data_->liquid_enthalpy[inlet_idx] = data_->inlet_enthalpy;
    data_->mixture_enthalpy[inlet_idx] = data_->inlet_enthalpy;
    data_->liquid_density[inlet_idx] = properties_->getLiquidDensity();
    data_->void_fraction[inlet_idx] = 0.0;
    data_->flow_quality[inlet_idx] = 0.0;
    data_->pressure[inlet_idx] = data_->inlet_pressure;

    // Calculate inlet momentum flux using APEX formulation for single-phase liquid
    // For single phase: spv = 1/ρₗ, so gv = (G²/ρₗ)
    double inlet_mass_flux = data_->mass_flow_rate / data_->flow_area;
    double inlet_liquid_density = properties_->getLiquidDensity();
    data_->momentum_flux[inlet_idx] = inlet_mass_flux * inlet_mass_flux / inlet_liquid_density;

    // Debug inlet conditions for first channel
    if (channel_id == 0) {
        double tsat = properties_->saturatedTemperature(data_->inlet_pressure);
        double inlet_temp = properties_->liquidTemperature(data_->inlet_pressure, data_->inlet_enthalpy);
        std::cout << "DEBUG Inlet: P=" << data_->inlet_pressure/1e6 << " MPa, Tsat=" << tsat-273.15
                  << "°C, Tinlet=" << inlet_temp-273.15 << "°C, h=" << data_->inlet_enthalpy/1e3
                  << " kJ/kg, Target T=278°C" << std::endl;

        // Test: calculate what enthalpy should be for 278°C at this pressure
        double target_h = properties_->liquidEnthalpy(278.0 + 273.15, data_->inlet_pressure);
        std::cout << "DEBUG Expected h for 278°C: " << target_h/1e3 << " kJ/kg" << std::endl;
    }

    // March from inlet to outlet
    for (int k = 1; k <= n_axial; ++k) {
        int prev_idx = getAxialIndex(channel_id, k-1);
        int curr_idx = getAxialIndex(channel_id, k);
        int cell_idx = getCellIndex(channel_id, k-1);

        // Calculate boiling/evaporation for this cell
        calculateBoiling(channel_id, k-1);

        // Solve enthalpy/mass balance equations
        double inlet_wl = data_->liquid_mass_flow[prev_idx];
        double inlet_wv = data_->vapor_mass_flow[prev_idx];
        double inlet_hm = data_->mixture_enthalpy[prev_idx];
        double heat_input = data_->heat_flux[cell_idx] * data_->heated_perimeter * data_->node_height;

        // Debug output for first node of selected channels
        if (k == 1 && (channel_id == 0 || channel_id == 4)) {
            std::cout << "  Channel " << channel_id << ": heat_flux[" << cell_idx << "] = "
                      << data_->heat_flux[cell_idx] << " W/m², heat_input = " << heat_input << " W" << std::endl;
        }

        double outlet_wl, outlet_wv, outlet_hm, outlet_hl, outlet_x, outlet_rhol;

        solveEnthalpyEquation(channel_id, k-1, inlet_wl, inlet_wv, inlet_hm, heat_input,
                             outlet_wl, outlet_wv, outlet_hm, outlet_hl, outlet_x, outlet_rhol);

        // Store results
        data_->liquid_mass_flow[curr_idx] = outlet_wl;
        data_->vapor_mass_flow[curr_idx] = outlet_wv;
        data_->total_mass_flow[curr_idx] = outlet_wl + outlet_wv;
        data_->mixture_enthalpy[curr_idx] = outlet_hm;
        data_->liquid_enthalpy[curr_idx] = outlet_hl;
        data_->flow_quality[curr_idx] = outlet_x;
        data_->liquid_density[curr_idx] = outlet_rhol;

        // Solve for void fraction and track iterations
        double void_frac;
        int newton_iters_before = getNewtonIterationCount();
        solveVoidFractionEquation(channel_id, k-1, outlet_wl, outlet_wv, outlet_rhol, void_frac);
        int newton_iters_used = getNewtonIterationCount() - newton_iters_before;

        // Update statistics (only for first channel to avoid spam)
        if (channel_id == 0) {
            total_newton_iterations += newton_iters_used;
            max_newton_iterations = std::max(max_newton_iterations, newton_iters_used);
        }

        data_->void_fraction[curr_idx] = void_frac;

        // Calculate momentum flux using APEX formulation
        // spv = (1-x)²/((1-α)ρₗ) + x²/(αρₘ)

        double specific_volume_momentum;
        if (outlet_x < 1e-10 && void_frac < 1e-6) {  // Single-phase liquid flow
            // Simplified single-phase momentum calculation
            specific_volume_momentum = 1.0 / outlet_rhol;
        } else {
            // Two-phase flow - full APEX formulation
            specific_volume_momentum = (1.0 - outlet_x) * (1.0 - outlet_x) / ((1.0 - void_frac) * outlet_rhol) +
                                     outlet_x * outlet_x / (void_frac * properties_->getVaporDensity() + 1e-20);
        }

        double total_mass_flux = (outlet_wl + outlet_wv) / data_->flow_area;
        data_->momentum_flux[curr_idx] = specific_volume_momentum * total_mass_flux * total_mass_flux;

        // Calculate pressure drop
        double pressure_drop;
        calculateAxialPressureDrop(channel_id, k-1, outlet_wl, outlet_wv, void_frac,
                                  outlet_rhol, outlet_x,
                                  data_->momentum_flux[curr_idx],
                                  data_->momentum_flux[prev_idx],
                                  0.0, pressure_drop); // No crossflow momentum for single channel

        // Apply pressure drop with safeguard against negative pressure
        double new_pressure = data_->pressure[prev_idx] - pressure_drop;

        // Clamp pressure to minimum realistic value (0.1 MPa = 100 kPa)
        data_->pressure[curr_idx] = std::max(new_pressure, 100000.0);
    }

    // Print Newton iteration statistics for first channel
    if (channel_id == 0) {
        double avg_newton_iterations = static_cast<double>(total_newton_iterations) / n_axial;
        std::cout << "Newton iteration statistics:" << std::endl;
        std::cout << "  Total Newton iterations: " << total_newton_iterations << std::endl;
        std::cout << "  Average per axial node: " << std::fixed << std::setprecision(1)
                  << avg_newton_iterations << std::endl;
        std::cout << "  Maximum for single node: " << max_newton_iterations << std::endl;
        std::cout << std::fixed << std::setprecision(4); // Reset precision
    }
}

void ThermalHydraulics::solveCrossflowIteration() {
    // ANTS Theory: Solve transverse momentum equation for all surfaces
    // d(G_m_CF × V_m_star × S_ns)/dz = (S_ns/ell) × (ΔP_ns - F_ns_dblprime)

    if (data_->getNumSurfaces() == 0) {
        return; // No surfaces to solve
    }

    int n_surfaces = data_->getNumSurfaces();
    int n_axial = data_->getNumAxialNodes();

    // Solve transverse momentum for each axial plane
    for (int k = 0; k <= n_axial; ++k) {
        solveCrossflowAtAxialPlane(k);
    }
}

void ThermalHydraulics::solveCrossflowAtAxialPlane(int axial_node) {
    // ANTS Theory: Solve nonlinear system of transverse momentum equations
    // For each surface: d(G_m_CF × V_m_star × S_ns)/dz = (S_ns/ell) × (ΔP_ns - F_ns_dblprime)

    int n_surfaces = data_->getNumSurfaces();
    if (n_surfaces == 0) return;

    // Build solution vectors
    std::vector<double> G_CF(n_surfaces);  // Crossflow mass flux for each surface
    std::vector<double> residual(n_surfaces);
    std::vector<std::vector<double>> jacobian(n_surfaces, std::vector<double>(n_surfaces));

    // Initialize crossflow from previous iteration or zero
    for (int s = 0; s < n_surfaces; ++s) {
        int surface_idx = getSurfaceIndex(s, axial_node);
        G_CF[s] = data_->crossflow_mass_flux[surface_idx];
    }

    // Newton-Raphson iteration for transverse momentum system
    bool converged = false;
    int max_inner_crossflow_iters = 10;
    double crossflow_tolerance = 1.0e-6;

    for (int iter = 0; iter < max_inner_crossflow_iters; ++iter) {
        // Calculate residuals for all surfaces
        calculateTransverseMomentumResiduals(axial_node, G_CF, residual);

        // Check convergence
        double max_residual = 0.0;
        for (int s = 0; s < n_surfaces; ++s) {
            max_residual = std::max(max_residual, std::abs(residual[s]));
        }

        if (max_residual < crossflow_tolerance) {
            converged = true;
            break;
        }

        // Build Jacobian matrix
        buildTransverseMomentumJacobian(axial_node, G_CF, jacobian);

        // Solve linear system: J * δG = -R
        std::vector<double> delta_G(n_surfaces);
        solveLinearSystem(jacobian, residual, delta_G);

        // Line search update with relaxation
        double lambda = 1.0;
        for (int s = 0; s < n_surfaces; ++s) {
            G_CF[s] -= lambda * delta_G[s];
        }

        // Apply bounds to crossflow (prevent excessive values)
        for (int s = 0; s < n_surfaces; ++s) {
            double max_crossflow = 1000.0; // kg/m²-s reasonable limit
            G_CF[s] = std::max(-max_crossflow, std::min(max_crossflow, G_CF[s]));
        }
    }

    // Store converged crossflow values
    for (int s = 0; s < n_surfaces; ++s) {
        int surface_idx = getSurfaceIndex(s, axial_node);
        data_->crossflow_mass_flux[surface_idx] = G_CF[s];
    }

    // Debug output for convergence issues
    if (!converged && axial_node == 0) {
        std::cout << "Warning: Crossflow did not converge at axial plane " << axial_node << std::endl;
    }
}

void ThermalHydraulics::calculateBoiling(int channel_id, int axial_node) {
    int cell_idx = getCellIndex(channel_id, axial_node);
    int prev_idx = getAxialIndex(channel_id, axial_node);

    // Get conditions at the upstream node
    double mass_flux = data_->total_mass_flow[prev_idx] / data_->flow_area;
    double heat_flux = data_->heat_flux[cell_idx];
    double liquid_enthalpy = data_->liquid_enthalpy[prev_idx];
    double void_fraction = data_->void_fraction[prev_idx];
    double liquid_temperature = properties_->liquidTemperature(data_->pressure[prev_idx],
                                                              liquid_enthalpy);

    // Calculate departure enthalpy (Saha-Zuber correlation)
    double departure_enthalpy = calculateDepartureEnthalpy(mass_flux, heat_flux);

    // Calculate subcooling
    double pressure = data_->pressure[prev_idx];
    double saturation_enthalpy = properties_->saturatedLiquidEnthalpy(pressure);
    double subcooling = saturation_enthalpy - liquid_enthalpy;

    // Debug output removed for cleaner results
    // if (channel_id == 0 && axial_node >= 22) {
    //     std::cout << "DEBUG Boiling node " << axial_node << ": T=" << liquid_temperature
    //               << "K, h=" << liquid_enthalpy << " J/kg, h_sat=" << saturation_enthalpy
    //               << " J/kg, subcooling=" << subcooling << " J/kg, departure_h=" << departure_enthalpy
    //               << " J/kg" << std::endl;
    // }

    // Calculate evaporation and condensation rates
    double evaporation_rate = 0.0;
    if (subcooling < departure_enthalpy) {
        if (subcooling <= 0.0) {
            // Superheated conditions - maximum evaporation based on available heat
            evaporation_rate = calculateEvaporationRate(heat_flux, 0.0, departure_enthalpy) * 10.0; // Boost for superheated
        } else {
            // Subcooled boiling - normal Saha-Zuber correlation
            evaporation_rate = calculateEvaporationRate(heat_flux, subcooling, departure_enthalpy);
        }
        // Debug output removed
        // if (channel_id == 0 && axial_node >= 22) {
        //     std::cout << "  Evaporation rate: " << evaporation_rate << " kg/m³-s" << std::endl;
        // }
    }

    double condensation_rate = calculateCondensationRate(void_fraction, subcooling);

    // Net evaporation rate
    double net_evaporation = std::max(0.0, evaporation_rate - condensation_rate);

    data_->evaporation_rate[cell_idx] = net_evaporation;
}

void ThermalHydraulics::solveEnthalpyEquation(int channel_id, int axial_node,
                                             double inlet_liquid_flow, double inlet_vapor_flow,
                                             double inlet_mixture_enthalpy, double heat_input,
                                             double& outlet_liquid_flow, double& outlet_vapor_flow,
                                             double& outlet_mixture_enthalpy, double& outlet_liquid_enthalpy,
                                             double& outlet_flow_quality, double& outlet_liquid_density) {

    int cell_idx = getCellIndex(channel_id, axial_node);
    double evaporation_rate = data_->evaporation_rate[cell_idx];
    double evaporation_mass = evaporation_rate * data_->flow_area * data_->node_height;

    // ANTS Theory discrete balance equations:
    // W_l,k = W_l,k-1 + dz_k × [-Γ' - sum_ns S_ns×(G_l_CF + G_l_TM + G_l_VD)]
    // W_v,k = W_v,k-1 + dz_k × [+Γ' - sum_ns S_ns×(G_v_CF + G_v_TM + G_v_VD)]

    // For single channel case, crossflow terms are zero
    outlet_liquid_flow = inlet_liquid_flow - evaporation_mass;
    outlet_vapor_flow = inlet_vapor_flow + evaporation_mass;

    // Ensure non-negative flows
    outlet_liquid_flow = std::max(0.0, outlet_liquid_flow);
    outlet_vapor_flow = std::max(0.0, outlet_vapor_flow);

    // ANTS Theory energy balance:
    // h_l,k = {(W_v,k-1 - W_v,k)×h_g + W_l,k-1×h_l,k-1 + dz_k×Q_wall_pp×P_H -
    //          dz_k×sum_ns S_ns×(G_m_CF×h_m_star + Q_m_TMpp + Q_m_VDpp)} / W_l,k

    double total_inlet_flow = inlet_liquid_flow + inlet_vapor_flow;
    double h_vapor = properties_->getVaporEnthalpy();

    // For single channel case, crossflow energy terms are zero
    // Energy from vapor generation/condensation
    double vapor_energy_change = (inlet_vapor_flow - outlet_vapor_flow) * h_vapor;

    // Total energy balance
    double total_energy_in = total_inlet_flow * inlet_mixture_enthalpy +
                           heat_input + vapor_energy_change;
    double total_flow_out = outlet_liquid_flow + outlet_vapor_flow;

    if (total_flow_out > EPS) {
        outlet_mixture_enthalpy = total_energy_in / total_flow_out;
    } else {
        outlet_mixture_enthalpy = inlet_mixture_enthalpy;
    }

    // Debug heat addition for first channel
    // Removed debug output for cleaner results

    // Calculate flow quality
    total_flow_out = outlet_liquid_flow + outlet_vapor_flow;
    if (total_flow_out > EPS) {
        outlet_flow_quality = outlet_vapor_flow / total_flow_out;
    } else {
        outlet_flow_quality = 0.0;
    }

    // Calculate liquid enthalpy from ANTS energy balance
    if (outlet_liquid_flow > EPS && outlet_flow_quality < 1.0) {
        // h_l,k = {total_energy - W_v,k × h_g} / W_l,k
        double liquid_energy = total_flow_out * outlet_mixture_enthalpy - outlet_vapor_flow * h_vapor;
        outlet_liquid_enthalpy = liquid_energy / outlet_liquid_flow;
    } else {
        outlet_liquid_enthalpy = properties_->getLiquidEnthalpy();
    }

    // Ensure liquid enthalpy doesn't exceed saturation
    double h_sat = properties_->getLiquidEnthalpy();
    if (outlet_liquid_enthalpy >= h_sat) {
        outlet_liquid_enthalpy = h_sat;
        outlet_flow_quality = (outlet_mixture_enthalpy - h_sat) /
                             (h_vapor - h_sat);
    }

    // Update liquid density
    outlet_liquid_density = properties_->densityFromEnthalpy(outlet_liquid_enthalpy);
}

void ThermalHydraulics::solveVoidFractionEquation(int channel_id, int axial_node,
                                                 double liquid_flow, double vapor_flow,
                                                 double liquid_density, double& void_fraction) {

    if (liquid_flow < EPS) {
        throw std::runtime_error("Negative liquid flow encountered");
    }

    double liquid_flux = liquid_flow / data_->flow_area;
    double vapor_flux = vapor_flow / data_->flow_area;

    // Initial guess
    void_fraction = std::max(0.0, vapor_flux / (liquid_flux + vapor_flux + EPS));

    // Track Newton-Raphson iterations
    int newton_iterations = 0;
    double initial_residual = 0.0;
    double final_residual = 0.0;
    bool converged = false;

    // Newton-Raphson iteration for drift-flux model
    for (int iter = 0; iter < data_->max_inner_iterations; ++iter) {
        newton_iterations = iter + 1;

        // Increment global Newton iteration counter
        incrementNewtonIterationCount();

        double C0 = calculateDistributionParameter(void_fraction);
        double Vgj = calculateDriftVelocity(void_fraction);
        double rho_vapor = properties_->getVaporDensity();

        // ANTS Theory: α = G_v / (C₀ × (G_v + (ρ_g/ρ_l) × G_l) + ρ_g × V_gj)
        // Rearranged as: f(α) = α × C₀ × (G_v + (ρ_g/ρ_l) × G_l) + α × ρ_g × V_gj - G_v = 0
        double total_flux = vapor_flux + (rho_vapor / liquid_density) * liquid_flux;
        double f0 = void_fraction * C0 * total_flux + void_fraction * rho_vapor * Vgj - vapor_flux;

        // Store residuals for reporting
        if (iter == 0) initial_residual = std::abs(f0);
        final_residual = std::abs(f0);

        if (std::abs(f0) < EPS) {
            converged = true;
            break;
        }

        // Numerical derivative
        double delta_alpha = 1.0e-4;
        double alpha1 = void_fraction + delta_alpha;
        double C0_1 = calculateDistributionParameter(alpha1);
        double Vgj_1 = calculateDriftVelocity(alpha1);

        double total_flux_1 = vapor_flux + (rho_vapor / liquid_density) * liquid_flux;
        double f1 = alpha1 * C0_1 * total_flux_1 + alpha1 * rho_vapor * Vgj_1 - vapor_flux;

        double dfda = (f1 - f0) / delta_alpha;

        double delta_void = -f0 / dfda;
        delta_void = std::max(delta_void, -0.1);
        delta_void = std::min(delta_void, 0.1);

        void_fraction += delta_void;
        void_fraction = std::max(0.0, std::min(void_fraction, 1.0 - EPS));
    }

    // Report convergence statistics for first few axial nodes or if not converged
    if ((channel_id == 0 && axial_node < 3) || !converged) {
        std::cout << "  Newton solve (Ch=" << channel_id << ", Node=" << axial_node
                  << "): " << newton_iterations << " iterations";
        if (converged) {
            std::cout << ", converged (residual: " << std::scientific << std::setprecision(2)
                      << initial_residual << " → " << final_residual << ")" << std::endl;
        } else {
            std::cout << ", NOT CONVERGED (residual: " << std::scientific << std::setprecision(2)
                      << final_residual << ")" << std::endl;
        }
        std::cout << std::fixed << std::setprecision(4); // Reset format
    }
}

void ThermalHydraulics::calculateAxialPressureDrop(int channel_id, int axial_node,
                                                  double liquid_flow, double vapor_flow,
                                                  double void_fraction, double liquid_density,
                                                  double flow_quality, double momentum_flux_out,
                                                  double momentum_flux_in, double crossflow_momentum,
                                                  double& pressure_drop) {

    double total_mass_flux = (liquid_flow + vapor_flow) / data_->flow_area;
    double rho_vapor = properties_->getVaporDensity();

    // Gravitational pressure drop
    double dp_gravity = GRAVITY_ACCEL * data_->node_height *
                       ((1.0 - void_fraction) * liquid_density + void_fraction * rho_vapor);

    // ANTS Theory two-phase pressure drop calculation
    // ΔP_2phi = f×(dz/D_H)×G_m²/(2×ρ_f)×φ_ch² + K×G_m²/(2×ρ_f)×φ_hom²

    // Reynolds number and friction factor (ANTS: f = 0.1892 × Re^(-0.2))
    double reynolds = calculateReynoldsNumber(total_mass_flux, data_->hydraulic_diameter, viscosity_liquid_);
    double friction_factor = 0.1892 * std::pow(reynolds + EPS, -0.2);  // ANTS specification

    // Calculate gamma parameter: γ = (ρ_f/ρ_g)^0.5 × (μ_g/μ_f)^0.2
    double mu_liquid = viscosity_liquid_;
    double mu_vapor = 1.0e-5;  // Approximate vapor viscosity
    double gamma = std::sqrt(liquid_density / rho_vapor) * std::pow(mu_vapor / mu_liquid, 0.2);
    double gamma_squared = gamma * gamma;

    // Calculate b parameter (piecewise function from ANTS theory)
    double b_param;
    if (gamma <= 9.5) {
        b_param = 55.0 / std::sqrt(total_mass_flux + EPS);
    } else if (gamma < 28.0) {
        b_param = 520.0 / (gamma * std::sqrt(total_mass_flux + EPS));
    } else {
        b_param = 15000.0 / (gamma_squared * std::sqrt(total_mass_flux + EPS));
    }

    // Chisholm multiplier: φ_ch² = 1 + (γ² - 1) × (b×X_f^0.9×(1-X_f)^0.9 + X_f^1.8)
    double x_09 = std::pow(flow_quality + EPS, 0.9);
    double x_18 = x_09 * x_09;
    double one_minus_x_09 = std::pow(1.0 - flow_quality + EPS, 0.9);

    double phi_ch_squared = 1.0 + (gamma_squared - 1.0) *
                           (b_param * x_09 * one_minus_x_09 + x_18);

    // Homogeneous multiplier: φ_hom² = 1 + X_f×(ρ_f/ρ_g - 1)
    double phi_hom_squared = 1.0 + flow_quality * (liquid_density / rho_vapor - 1.0);

    // Friction pressure drop: f×(dz/D_H)×G_m²/(2×ρ_f)×φ_ch²
    double dp_friction = friction_factor * (data_->node_height / data_->hydraulic_diameter) *
                        (total_mass_flux * total_mass_flux) / (2.0 * liquid_density) * phi_ch_squared;

    // Form loss pressure drop: K×G_m²/(2×ρ_f)×φ_hom²
    double dp_form = data_->loss_coefficient *
                    (total_mass_flux * total_mass_flux) / (2.0 * liquid_density) * phi_hom_squared;

    // Momentum pressure drop
    double dp_momentum = momentum_flux_out - momentum_flux_in;

    // Crossflow momentum contribution
    double dp_crossflow = crossflow_momentum / data_->flow_area;

    // Total pressure drop components
    pressure_drop = dp_gravity + dp_friction + dp_form + dp_momentum + dp_crossflow;
}

double ThermalHydraulics::calculateDepartureEnthalpy(double mass_flux, double heat_flux) const {
    // ANTS Theory: Void departure criterion from mechanistic subcooled boiling
    // h_f - h_l_d = Peclet-dependent expression
    double peclet = prandtl_number_ * mass_flux * data_->hydraulic_diameter / (viscosity_liquid_ + EPS);

    double departure_enthalpy;
    if (peclet < 70000.0) {
        // ANTS: (h_f - h_l_d) = 0.0022 × Pe × (Q_wall_pp / G_m)
        departure_enthalpy = 0.0022 * peclet * (heat_flux / (mass_flux + EPS));
    } else {
        // ANTS: (h_f - h_l_d) = 154 × (Q_wall_pp / G_m)
        departure_enthalpy = 154.0 * (heat_flux / (mass_flux + EPS));
    }

    return departure_enthalpy;
}

double ThermalHydraulics::calculateEvaporationRate(double heat_flux, double subcooling,
                                                  double departure_enthalpy) const {
    double h_fg = properties_->getEnthalpyOfVaporization();
    double rho_vapor = properties_->getVaporDensity();
    double rho_liquid = properties_->getLiquidDensity();

    // ANTS Theory mechanistic boiling model
    // Q_boil_pp = Q_wall_pp × [1 - (h_f - h_l)/(h_f - h_l_d)] when h_l > h_l_d
    double boiling_heat_flux;
    if (subcooling < departure_enthalpy) {
        // Activate boiling when liquid enthalpy exceeds departure enthalpy
        boiling_heat_flux = heat_flux * (1.0 - subcooling / (departure_enthalpy + EPS));
    } else {
        boiling_heat_flux = 0.0;  // No boiling in highly subcooled region
    }

    // Calculate pumping parameter: ε_pump = [ρ_l×(h_f - h_l)] / [ρ_g × h_fg]
    double epsilon_pump = rho_liquid * subcooling / (rho_vapor * h_fg);

    // Net evaporation rate: Γ' = (P_H × Q_boil_pp) / (h_fg × (1 + ε_pump))
    return boiling_heat_flux / (h_fg * (1.0 + epsilon_pump));
}

double ThermalHydraulics::calculateCondensationRate(double void_fraction, double subcooling) const {
    // ANTS Theory condensation model
    // P_H × γ_cond = H₀ × (1/v_fg) × A_f × α × (T_sat - T_l)
    // Where H₀ = 0.075 s⁻¹ K⁻¹ (ANTS constant)

    const double H0 = 0.075;  // s⁻¹ K⁻¹ from ANTS theory
    double h_fg = properties_->getEnthalpyOfVaporization();
    double rho_vapor = properties_->getVaporDensity();
    double v_fg = 1.0 / rho_vapor;  // Specific volume of vapor

    // Temperature difference: use subcooling directly (already in temperature units)
    double temp_diff = subcooling / 4200.0;  // Convert enthalpy subcooling to temperature

    // γ_cond = H₀ × (1/v_fg) × (A_f/P_H) × α × (T_sat - T_l)
    double condensation_rate = H0 * (1.0 / v_fg) * (data_->flow_area / data_->heated_perimeter) *
                              void_fraction * temp_diff;

    return std::max(0.0, condensation_rate);
}

double ThermalHydraulics::calculateHomogeneousMultiplier(double flow_quality,
                                                        double liquid_density, double vapor_density) const {
    return 1.0 + flow_quality * (liquid_density / vapor_density - 1.0);
}

double ThermalHydraulics::calculateChisholmMultiplier(double flow_quality, double mass_flux) const {
    if (flow_quality >= 1.0) {
        return 1.0 + gamma_squared_;
    }

    double b_param = chisholm_parameter_b_ / std::sqrt(mass_flux + EPS);
    double x_power = std::pow(flow_quality, 0.9);

    return 1.0 + gamma_squared_ * (b_param * x_power * std::pow(1.0 - flow_quality, 0.9) +
                                  x_power * x_power);
}

double ThermalHydraulics::calculateFrictionFactor(double reynolds) const {
    // Blasius correlation for smooth pipes
    return 0.1892 * std::pow(reynolds + EPS, -0.2);
}

double ThermalHydraulics::calculateDistributionParameter(double void_fraction) const {
    return properties_->distributionParameter(void_fraction);
}

double ThermalHydraulics::calculateDriftVelocity(double void_fraction) const {
    return properties_->driftVelocity(void_fraction);
}

bool ThermalHydraulics::checkConvergence() const {
    if (old_crossflow_.empty() || data_->crossflow_mass_flux.empty()) {
        return true; // Single channel case
    }

    double max_residual = 0.0;
    for (size_t i = 0; i < old_crossflow_.size(); ++i) {
        double residual = std::abs(data_->crossflow_mass_flux[i] - old_crossflow_[i]);
        max_residual = std::max(max_residual, residual);
    }

    return max_residual < data_->outer_tolerance;
}

double ThermalHydraulics::getExitVoidFraction(int channel_id) const {
    int n_axial = data_->getNumAxialNodes();
    int exit_idx = getAxialIndex(channel_id, n_axial);
    return data_->void_fraction[exit_idx];
}

double ThermalHydraulics::getExitTemperature(int channel_id) const {
    int n_axial = data_->getNumAxialNodes();
    int exit_idx = getAxialIndex(channel_id, n_axial);
    double exit_enthalpy = data_->liquid_enthalpy[exit_idx];
    double exit_temp_k = properties_->liquidTemperature(data_->pressure[exit_idx], exit_enthalpy);
    return exit_temp_k - 273.15; // Convert to Celsius
}

double ThermalHydraulics::getExitTemperatureK(int channel_id) const {
    int n_axial = data_->getNumAxialNodes();
    int exit_idx = getAxialIndex(channel_id, n_axial);
    double exit_enthalpy = data_->liquid_enthalpy[exit_idx];
    return properties_->liquidTemperature(data_->pressure[exit_idx], exit_enthalpy);
}

double ThermalHydraulics::getPressureDrop(int channel_id) const {
    int inlet_idx = getAxialIndex(channel_id, 0);
    int n_axial = data_->getNumAxialNodes();
    int exit_idx = getAxialIndex(channel_id, n_axial);
    return data_->pressure[inlet_idx] - data_->pressure[exit_idx];
}

void ThermalHydraulics::printSolution() const {
    std::cout << "\n=== ANTS Subchannel Solution ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    for (int ch = 0; ch < data_->getNumSubchannels(); ++ch) {
        printChannelResults(ch);
    }

    // Print summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total/Average Bundle Pressure Drop [Pa]: " << getPressureDrop(0) << std::endl;
    std::cout << "Exit Void Fraction [-]: " << getExitVoidFraction(0) << std::endl;
    std::cout << "Exit Temperature [°C]: " << getExitTemperature(0) << std::endl;
    std::cout << "Exit Temperature [K]: " << getExitTemperatureK(0) << std::endl;
}

void ThermalHydraulics::printChannelResults(int channel_id) const {
    std::cout << "\nChannel " << channel_id << " Results:" << std::endl;
    std::cout << "Axial Node | Void [-] | Quality [-] | Pressure [Pa] | Temperature [K]" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int k = 0; k <= data_->getNumAxialNodes(); ++k) {
        int idx = getAxialIndex(channel_id, k);
        double temp = properties_->liquidTemperature(data_->pressure[idx], data_->liquid_enthalpy[idx]);

        std::cout << std::setw(10) << k
                  << " | " << std::setw(8) << data_->void_fraction[idx]
                  << " | " << std::setw(11) << data_->flow_quality[idx]
                  << " | " << std::setw(13) << data_->pressure[idx]
                  << " | " << std::setw(14) << temp << std::endl;
    }
}

int ThermalHydraulics::getAxialIndex(int channel_id, int axial_node) const {
    return channel_id * (data_->getNumAxialNodes() + 1) + axial_node;
}

int ThermalHydraulics::getCellIndex(int channel_id, int axial_node) const {
    return channel_id * data_->getNumAxialNodes() + axial_node;
}

double ThermalHydraulics::calculateReynoldsNumber(double mass_flux, double hydraulic_diameter,
                                                 double viscosity) const {
    return mass_flux * hydraulic_diameter / (viscosity + EPS);
}

int ThermalHydraulics::getNewtonIterationCount() const {
    return global_newton_iteration_count;
}

void ThermalHydraulics::incrementNewtonIterationCount() {
    global_newton_iteration_count++;
}

void ThermalHydraulics::resetNewtonIterationCount() {
    global_newton_iteration_count = 0;
}

void ThermalHydraulics::calculateTransverseMomentumResiduals(int axial_node,
                                                           const std::vector<double>& G_CF,
                                                           std::vector<double>& residual) {
    // ANTS Theory: Calculate residuals for transverse momentum equation
    // d(G_m_CF × V_m_star × S_ns)/dz = (S_ns/ell) × (ΔP_ns - F_ns_dblprime)

    int n_surfaces = data_->getNumSurfaces();

    for (int s = 0; s < n_surfaces; ++s) {
        // Get surface geometry
        double S_ns = data_->gap_width;  // Gap width
        double ell = data_->gap_spacing; // Transverse path length

        // Get adjacent channel indices for this surface
        int ch_i, ch_j;
        getSurfaceChannels(s, ch_i, ch_j);

        // Get pressures at adjacent channels
        int idx_i = getAxialIndex(ch_i, axial_node);
        int idx_j = getAxialIndex(ch_j, axial_node);
        double P_i = data_->pressure[idx_i];
        double P_j = data_->pressure[idx_j];
        double DeltaP_ns = P_i - P_j;

        // Calculate mixture properties for form loss
        double void_i = data_->void_fraction[idx_i];
        double void_j = data_->void_fraction[idx_j];
        double rho_l_i = data_->liquid_density[idx_i];
        double rho_l_j = data_->liquid_density[idx_j];
        double rho_g = properties_->getVaporDensity();

        // Mixture density (star denotes donor cell evaluation)
        double rho_m_star;
        if (G_CF[s] >= 0.0) {
            // Flow from i to j, use donor cell i
            rho_m_star = (1.0 - void_i) * rho_l_i + void_i * rho_g;
        } else {
            // Flow from j to i, use donor cell j
            rho_m_star = (1.0 - void_j) * rho_l_j + void_j * rho_g;
        }

        // Form loss coefficient (gap loss)
        double K_ns = data_->loss_coefficient;

        // Form loss pressure drop: F_ns_dblprime = 0.5 × K_ns × (G_m_CF × |G_m_CF|) / rho_m_star
        double F_ns_dblprime = 0.5 * K_ns * G_CF[s] * std::abs(G_CF[s]) / (rho_m_star + EPS);

        // Momentum flux derivative (simplified for axial differences)
        double V_m_star = calculateMixtureVelocity(s, axial_node, G_CF[s]);
        double momentum_change = 0.0;

        if (axial_node > 0) {
            // Calculate axial derivative d(G_m_CF × V_m_star × S_ns)/dz
            int prev_surface_idx = getSurfaceIndex(s, axial_node - 1);
            double G_CF_prev = data_->crossflow_mass_flux[prev_surface_idx];
            double V_m_prev = calculateMixtureVelocity(s, axial_node - 1, G_CF_prev);

            momentum_change = (G_CF[s] * V_m_star * S_ns - G_CF_prev * V_m_prev * S_ns) / data_->node_height;
        }

        // ANTS transverse momentum residual
        // R = d(G_m_CF × V_m_star × S_ns)/dz - (S_ns/ell) × (ΔP_ns - F_ns_dblprime)
        residual[s] = momentum_change - (S_ns / ell) * (DeltaP_ns - F_ns_dblprime);
    }
}

void ThermalHydraulics::buildTransverseMomentumJacobian(int axial_node,
                                                      const std::vector<double>& G_CF,
                                                      std::vector<std::vector<double>>& jacobian) {
    // ANTS Theory: Build Jacobian matrix for Newton-Raphson solution
    // J[i][j] = ∂R_i/∂G_CF_j

    int n_surfaces = data_->getNumSurfaces();
    double perturbation = 1.0e-6;

    // Initialize Jacobian
    for (int i = 0; i < n_surfaces; ++i) {
        for (int j = 0; j < n_surfaces; ++j) {
            jacobian[i][j] = 0.0;
        }
    }

    // Calculate baseline residuals
    std::vector<double> residual_base(n_surfaces);
    calculateTransverseMomentumResiduals(axial_node, G_CF, residual_base);

    // Numerical differentiation for Jacobian
    for (int j = 0; j < n_surfaces; ++j) {
        std::vector<double> G_CF_pert = G_CF;
        G_CF_pert[j] += perturbation;

        std::vector<double> residual_pert(n_surfaces);
        calculateTransverseMomentumResiduals(axial_node, G_CF_pert, residual_pert);

        for (int i = 0; i < n_surfaces; ++i) {
            jacobian[i][j] = (residual_pert[i] - residual_base[i]) / perturbation;
        }
    }
}

double ThermalHydraulics::calculateMixtureVelocity(int surface_id, int axial_node, double G_CF) {
    // Calculate mixture velocity for crossflow momentum calculation
    // This is a simplified implementation - full version would use proper two-phase velocity

    // Get adjacent channels
    int ch_i, ch_j;
    getSurfaceChannels(surface_id, ch_i, ch_j);

    // Use donor cell properties
    int donor_channel = (G_CF >= 0.0) ? ch_i : ch_j;
    int idx = getAxialIndex(donor_channel, axial_node);

    double rho_mixture = (1.0 - data_->void_fraction[idx]) * data_->liquid_density[idx] +
                        data_->void_fraction[idx] * properties_->getVaporDensity();

    // Simple velocity estimate: V = G / ρ
    return std::abs(G_CF) / (rho_mixture + EPS);
}

void ThermalHydraulics::getSurfaceChannels(int surface_id, int& ch_i, int& ch_j) {
    // Map surface ID to adjacent channel IDs
    // This is a simplified mapping - full implementation would use connectivity data

    // For 3x3 geometry with 12 surfaces, map to adjacent channels
    // Surface connectivity for 3x3 pin-centered subchannels
    const int surface_connectivity[][2] = {
        {0, 1}, {1, 2},           // Row 1 horizontal
        {3, 4}, {4, 5},           // Row 2 horizontal
        {6, 7}, {7, 8},           // Row 3 horizontal
        {0, 3}, {1, 4}, {2, 5},   // Vertical connections
        {3, 6}, {4, 7}, {5, 8}    // Vertical connections
    };

    if (surface_id < 12) {
        ch_i = surface_connectivity[surface_id][0];
        ch_j = surface_connectivity[surface_id][1];
    } else {
        // Default fallback
        ch_i = 0;
        ch_j = 1;
    }
}

void ThermalHydraulics::solveLinearSystem(const std::vector<std::vector<double>>& A,
                                        const std::vector<double>& b,
                                        std::vector<double>& x) {
    // Solve Ax = b using Gaussian elimination with partial pivoting
    int n = A.size();

    // Create augmented matrix
    std::vector<std::vector<double>> aug(n, std::vector<double>(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i][j] = A[i][j];
        }
        aug[i][n] = -b[i]; // Note: we want to solve J * δG = -R
    }

    // Gaussian elimination with partial pivoting
    for (int k = 0; k < n - 1; ++k) {
        // Find pivot
        int pivot_row = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(aug[i][k]) > std::abs(aug[pivot_row][k])) {
                pivot_row = i;
            }
        }

        // Swap rows
        if (pivot_row != k) {
            aug[k].swap(aug[pivot_row]);
        }

        // Eliminate
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(aug[k][k]) > EPS) {
                double factor = aug[i][k] / aug[k][k];
                for (int j = k; j <= n; ++j) {
                    aug[i][j] -= factor * aug[k][j];
                }
            }
        }
    }

    // Back substitution
    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = aug[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= aug[i][j] * x[j];
        }
        if (std::abs(aug[i][i]) > EPS) {
            x[i] /= aug[i][i];
        } else {
            x[i] = 0.0; // Singular matrix handling
        }
    }
}

int ThermalHydraulics::getSurfaceIndex(int surface_id, int axial_node) const {
    // Calculate linear index for surface data arrays
    return surface_id * (data_->getNumAxialNodes() + 1) + axial_node;
}

} // namespace subchannel
} // namespace ants
