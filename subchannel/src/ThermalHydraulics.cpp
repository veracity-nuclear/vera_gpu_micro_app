#include "ThermalHydraulics.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <algorithm>

namespace ants {
namespace subchannel {

ThermalHydraulics::ThermalHydraulics(std::shared_ptr<SubchannelData> data,
                                   std::shared_ptr<Properties> properties)
    : data_(data)
    , properties_(properties)
    , viscosity_liquid_(0.0001)          // [Pa-s]
    , thermal_conductivity_(0.6)         // [W/m-K]  
    , prandtl_number_(1.0)
    , reynolds_number_(10000.0)
    , peclet_number_(10000.0)
    , chisholm_parameter_b_(0.25)        // Chisholm correlation parameter
    , gamma_squared_(1.0)                // Two-phase multiplier parameter
{
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
    
    if (n_channels == 1) {
        solveSingleChannel(0);
    } else {
        solveMultiChannel();
    }
}

void ThermalHydraulics::solveSingleChannel(int channel_id) {
    // For single channel, just perform axial march
    solveAxialMarch(channel_id);
}

void ThermalHydraulics::solveMultiChannel() {
    // Outer iteration loop for crossflow convergence
    int outer_iter = 0;
    bool converged = false;
    
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
        ++outer_iter;
    }
    
    if (!converged) {
        std::cout << "Warning: Outer iteration did not converge after " 
                  << outer_iter << " iterations" << std::endl;
    }
}

void ThermalHydraulics::solveAxialMarch(int channel_id) {
    int n_axial = data_->getNumAxialNodes();
    
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
        
        // Solve for void fraction
        double void_frac;
        solveVoidFractionEquation(channel_id, k-1, outlet_wl, outlet_wv, outlet_rhol, void_frac);
        data_->void_fraction[curr_idx] = void_frac;
        
        // Calculate momentum flux
        double specific_volume = (1.0 - outlet_x) / outlet_rhol + 
                               outlet_x / properties_->getVaporDensity();
        double total_mass_flux = (outlet_wl + outlet_wv) / data_->flow_area;
        data_->momentum_flux[curr_idx] = specific_volume * total_mass_flux * total_mass_flux;
        
        // Calculate pressure drop
        double pressure_drop;
        calculateAxialPressureDrop(channel_id, k-1, outlet_wl, outlet_wv, void_frac, 
                                  outlet_rhol, outlet_x,
                                  data_->momentum_flux[curr_idx], 
                                  data_->momentum_flux[prev_idx],
                                  0.0, pressure_drop); // No crossflow momentum for single channel
        
        data_->pressure[curr_idx] = data_->pressure[prev_idx] - pressure_drop;
    }
}

void ThermalHydraulics::solveCrossflowIteration() {
    // For now, implement a simple pressure-driven crossflow model
    // In full implementation, this would solve the transverse momentum equation
    
    // This is a placeholder - full crossflow solver would be implemented here
    // For single channel problems, this does nothing
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
    double saturation_enthalpy = properties_->getLiquidEnthalpy();
    double subcooling = saturation_enthalpy - liquid_enthalpy;
    
    // Calculate evaporation and condensation rates
    double evaporation_rate = 0.0;
    if (subcooling < departure_enthalpy && subcooling > 0.0) {
        evaporation_rate = calculateEvaporationRate(heat_flux, subcooling, departure_enthalpy);
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
    double evaporation_mass = evaporation_rate * data_->flow_area;
    
    // Mass balance (assuming no crossflow for now)
    double inlet_total_flow = inlet_liquid_flow + inlet_vapor_flow;
    outlet_vapor_flow = std::max(0.0, inlet_vapor_flow + evaporation_mass);
    outlet_liquid_flow = inlet_total_flow - outlet_vapor_flow;
    
    // Energy balance
    outlet_mixture_enthalpy = (inlet_total_flow * inlet_mixture_enthalpy + heat_input) / 
                             inlet_total_flow;
    
    // Calculate flow quality
    if (inlet_total_flow > EPS) {
        outlet_flow_quality = outlet_vapor_flow / inlet_total_flow;
    } else {
        outlet_flow_quality = 0.0;
    }
    
    // Calculate liquid enthalpy
    double h_vapor = properties_->getVaporEnthalpy();
    if (outlet_flow_quality < 1.0) {
        outlet_liquid_enthalpy = (outlet_mixture_enthalpy - outlet_flow_quality * h_vapor) /
                                (1.0 - outlet_flow_quality);
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
    
    // Newton-Raphson iteration for drift-flux model
    for (int iter = 0; iter < data_->max_inner_iterations; ++iter) {
        double C0 = calculateDistributionParameter(void_fraction);
        double Vgj = calculateDriftVelocity(void_fraction);
        double rho_vapor = properties_->getVaporDensity();
        
        double f0 = void_fraction * C0 * (rho_vapor / liquid_density * liquid_flux + vapor_flux) +
                   void_fraction * rho_vapor * Vgj - vapor_flux;
        
        if (std::abs(f0) < EPS) {
            break;
        }
        
        // Numerical derivative
        double delta_alpha = 1.0e-4;
        double alpha1 = void_fraction + delta_alpha;
        double C0_1 = calculateDistributionParameter(alpha1);  
        double Vgj_1 = calculateDriftVelocity(alpha1);
        
        double f1 = alpha1 * C0_1 * (rho_vapor / liquid_density * liquid_flux + vapor_flux) +
                   alpha1 * rho_vapor * Vgj_1 - vapor_flux;
                   
        double dfda = (f1 - f0) / delta_alpha;
        
        double delta_void = -f0 / dfda;
        delta_void = std::max(delta_void, -0.1);
        delta_void = std::min(delta_void, 0.1);
        
        void_fraction += delta_void;
        void_fraction = std::max(0.0, std::min(void_fraction, 1.0 - EPS));
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
    
    // Friction pressure drop with two-phase multiplier
    double homogeneous_multiplier = calculateHomogeneousMultiplier(flow_quality, liquid_density, rho_vapor);
    double chisholm_multiplier = calculateChisholmMultiplier(flow_quality, total_mass_flux);
    
    // Reynolds number and friction factor
    double reynolds = calculateReynoldsNumber(total_mass_flux, data_->hydraulic_diameter, viscosity_liquid_);
    double friction_factor = calculateFrictionFactor(reynolds);
    double loss_coeff = friction_factor * data_->node_height / data_->hydraulic_diameter * chisholm_multiplier +
                       data_->loss_coefficient * homogeneous_multiplier;
    
    double dp_friction = 0.5 * loss_coeff * total_mass_flux * total_mass_flux / liquid_density;
    
    // Momentum pressure drop  
    double dp_momentum = momentum_flux_out - momentum_flux_in;
    
    // Crossflow momentum contribution
    double dp_crossflow = crossflow_momentum / data_->flow_area;
    
    pressure_drop = dp_gravity + dp_friction + dp_momentum + dp_crossflow;
}

double ThermalHydraulics::calculateDepartureEnthalpy(double mass_flux, double heat_flux) const {
    // Saha-Zuber correlation for onset of nucleate boiling
    double peclet = prandtl_number_ * mass_flux * data_->hydraulic_diameter / viscosity_liquid_;
    
    double departure_enthalpy;
    if (peclet < 70000.0) {
        departure_enthalpy = 0.0022 * peclet * heat_flux / (mass_flux + EPS);
    } else {
        departure_enthalpy = 154.0 * heat_flux / (mass_flux + EPS);
    }
    
    return departure_enthalpy;
}

double ThermalHydraulics::calculateEvaporationRate(double heat_flux, double subcooling, 
                                                  double departure_enthalpy) const {
    double h_fg = properties_->getEnthalpyOfVaporization();
    double rho_vapor = properties_->getVaporDensity();
    double rho_liquid = properties_->getLiquidDensity();
    
    double boiling_fraction;
    if (subcooling < departure_enthalpy) {
        boiling_fraction = 1.0 - subcooling / departure_enthalpy;
    } else {
        boiling_fraction = 0.0;
    }
    
    double evaporation_flux = boiling_fraction * heat_flux;
    double subcooling_factor = rho_liquid * subcooling / (rho_vapor * h_fg);
    
    return evaporation_flux / (h_fg * (1.0 + subcooling_factor));
}

double ThermalHydraulics::calculateCondensationRate(double void_fraction, double subcooling) const {
    // Simplified condensation model (Lahey correlation)
    double t_sat = properties_->getSaturationTemperature();
    double liquid_temp = t_sat - subcooling / 4200.0; // Approximate cp
    
    double condensation_rate = data_->node_height * CONDENSATION_COEFF * 
                              properties_->getLiquidDensity() * properties_->getEnthalpyOfVaporization() *
                              void_fraction * (t_sat - liquid_temp) / 
                              (data_->flow_area * properties_->getEnthalpyOfVaporization());
                              
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

} // namespace subchannel
} // namespace ants
