#include "Properties.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace ants {
namespace subchannel {

Properties::Properties()
    : n_alpha_pts_(0), d_alpha_(100.0), alpha_0_(0.0)
    , n_enthalpy_pts_(0), d_enthalpy_(0.02), h_0_(850.0)
    , n_temp_pts_(0), d_temp_(0.02), t_0_(0.0)
    , operating_pressure_(7.255e6)  // Default 7.255 MPa
    , t_sat_(558.0)
    , rho_liquid_(750.0)
    , rho_vapor_(40.0)
    , h_liquid_(1200000.0)
    , h_vapor_(2800000.0)
    , h_fg_(1600000.0)
{
    buildSteamTables();
    // Don't call setOperatingPressure here since it may fail with default values
    // It will be called explicitly later when the correct pressure is set
}

void Properties::initialize() {
    buildSteamTables();
    // Don't automatically set operating pressure - will be set explicitly
}

void Properties::buildSteamTables() {
    // Initialize table sizes - these would be read from data files in full implementation
    n_alpha_pts_ = 100;
    n_enthalpy_pts_ = 200;
    n_temp_pts_ = 200;

    // Resize tables
    c0_table_.resize(n_alpha_pts_ + 1);
    vg_table_.resize(n_alpha_pts_ + 1);
    rho_h_table_.resize(n_enthalpy_pts_ + 1);
    t_h_table_.resize(n_enthalpy_pts_ + 1);
    rho_t_table_.resize(n_temp_pts_ + 1);
    h_t_table_.resize(n_temp_pts_ + 1);

    // For now, initialize with simple correlations
    // In full implementation, these would be loaded from property files

    // Initialize void fraction tables with full Chexal-Lellouche correlations
    // Following ANTS theory implementation
    double rhof = 750.0;  // kg/m³ - liquid density
    double rhog = 40.0;   // kg/m³ - vapor density
    double pc = 22.12e6;   // Pa - critical pressure (22.12 MPa for water)

    // Chexal-Lellouche correlation parameters (ANTS theory constants)
    double B1 = ANTS_B1;   // ANTS constant (1.5)
    double B2 = ANTS_B2;   // ANTS constant (1.41)

    // Calculate K0 parameter: K₀ = B₁ + (1-B₁) × (ρ_g/ρ_f)^0.25
    double K0 = B1 + (1.0 - B1) * std::pow(rhog/rhof, 0.25);

    // Calculate C1 parameter: C₁ = 4×P²/(P_crit×(P_crit - P))
    double C1 = 4.0 * operating_pressure_ * operating_pressure_ /
                (pc * (pc - operating_pressure_));

    // Surface tension and gravity for drift velocity
    double sigma0 = 0.072;  // N/m - surface tension for water
    double ggrav = GRAVITY_ACCEL;

    // V_gj0 = B₂ × ((ρ_f - ρ_g) × σ × g / ρ_f²)^0.25
    double vgj0 = B2 * std::pow(sigma0 * ggrav * (rhof - rhog) / (rhof * rhof), 0.25);

    for (int i = 0; i <= n_alpha_pts_; ++i) {
        double alpha = alpha_0_ + i / d_alpha_;
        alpha = std::min(alpha, 0.999);

        // Calculate L factor: L = (1 - exp(-C₁×α)) / (1 - exp(-C₁))
        double L;
        if (std::abs(C1) < 1e-10) {
            L = alpha;  // Limit as C1 → 0
        } else {
            double exp_c1_alpha = std::exp(-C1 * alpha);
            double exp_c1 = std::exp(-C1);
            L = (1.0 - exp_c1_alpha) / (1.0 - exp_c1);
        }

        // For now, use simple reference void fraction (could be made pressure-dependent)
        double alpha_r = 0.8;  // Typical reference void fraction

        // Distribution parameter: C₀ = L / (K₀ + (1-K₀) × α_r)
        c0_table_[i] = L / (K0 + (1.0 - K0) * alpha_r);

        // Drift velocity: V_gj = V_gj0 × (1-α) × B₁
        vg_table_[i] = vgj0 * (1.0 - alpha) * B1;
    }

    // Initialize temperature-enthalpy table (simplified liquid water properties)
    // Set up realistic temperature range: 250K to 650K (covers liquid water range)
    double temp_min = 250.0;  // K
    double temp_max = 650.0;  // K
    double temp_range = temp_max - temp_min;

    for (int i = 0; i <= n_temp_pts_; ++i) {
        double temperature = temp_min + (temp_range * i) / n_temp_pts_;

        // Simple enthalpy correlation for liquid water (approximately 4.18 kJ/kg-K specific heat)
        // h = cp * (T - T_ref), using T_ref = 273.15 K (0°C)
        double enthalpy = 4180.0 * (temperature - 273.15);
        h_t_table_[i] = std::max(enthalpy, 0.0);  // Ensure non-negative

        // Simple density correlation for liquid water
        rho_t_table_[i] = 1000.0 * (1.0 - 2.1e-4 * std::pow(temperature - 277.0, 2.0) / 1000.0);
        rho_t_table_[i] = std::max(rho_t_table_[i], 700.0); // Ensure reasonable bounds
    }

    // Update table parameters to match the actual range used
    t_0_ = temp_min;  // Start temperature for table
    d_temp_ = n_temp_pts_ / temp_range;  // Points per Kelvin

    // Initialize enthalpy-based tables
    for (int i = 0; i <= n_enthalpy_pts_; ++i) {
        double enthalpy = h_0_ + i / d_enthalpy_;

        // Simple inverse correlation: T = T_ref + h/cp
        double temperature = 273.15 + enthalpy / 4180.0;
        t_h_table_[i] = temperature;

        // Corresponding density
        rho_h_table_[i] = 1000.0 * (1.0 - 2.1e-4 * std::pow(temperature - 277.0, 2.0) / 1000.0);
        rho_h_table_[i] = std::max(rho_h_table_[i], 700.0);
    }
}

void Properties::setOperatingPressure(double pressure) {
    operating_pressure_ = pressure;
    t_sat_ = saturatedTemperature(pressure);
    rho_liquid_ = saturatedLiquidDensity(t_sat_);
    rho_vapor_ = saturatedVaporDensity(t_sat_);
    h_liquid_ = saturatedLiquidEnthalpy(pressure);
    h_vapor_ = saturatedVaporEnthalpy(pressure);
    h_fg_ = h_vapor_ - h_liquid_;
}

double Properties::saturatedTemperature(double pressure) const {
    // More accurate saturation temperature correlation for water
    if (pressure <= 0.0) {
        std::cout << "ERROR: Invalid pressure for saturation temperature: " << pressure << " Pa" << std::endl;
        if (std::isnan(pressure)) std::cout << "  Pressure is NaN" << std::endl;
        if (std::isinf(pressure)) std::cout << "  Pressure is infinite" << std::endl;
        throw std::runtime_error("Invalid pressure for saturation temperature calculation");
    }

    // Improved correlation for water saturation temperature
    // Based on Wagner-Pruss equation (simplified)
    double p_bar = pressure / 1.0e5;  // Convert Pa to bar

    if (p_bar < 1.0) {
        // Low pressure approximation
        return 273.15 + 100.0 + 30.0 * std::log(p_bar);
    } else {
        // High pressure approximation (better for PWR conditions)
        // Empirical fit to steam table data for 1-100 bar range
        double ln_p = std::log(p_bar);
        return 273.15 + 100.0 + 32.55 * ln_p + 0.7 * ln_p * ln_p;
    }
}

double Properties::saturatedPressure(double temperature) const {
    // Antoine equation approximation
    if (temperature <= 273.15) {
        throw std::runtime_error("Temperature below freezing point");
    }

    double t_c = temperature - 273.15;
    return 101325.0 * std::exp(0.074 * t_c - 0.0002 * t_c * t_c);
}

double Properties::saturatedPressureDerivative(double temperature) const {
    double psat = saturatedPressure(temperature);
    double t_c = temperature - 273.15;
    return psat * (0.074 - 0.0004 * t_c);
}

double Properties::saturatedLiquidDensity(double temperature) const {
    // Simplified saturated liquid density correlation
    if (temperature >= TC) {
        return RHOC;
    }

    double tau = 1.0 - temperature / TC;
    return RHOC * (1.0 + 1.99 * std::pow(tau, 0.35) - 0.09 * tau);
}

double Properties::saturatedVaporDensity(double temperature) const {
    double psat = saturatedPressure(temperature);
    return waterDensity(temperature, psat, 0);
}

double Properties::saturatedLiquidEnthalpy(double pressure) const {
    double tsat = saturatedTemperature(pressure);
    return liquidEnthalpy(tsat, pressure);
}

double Properties::saturatedVaporEnthalpy(double pressure) const {
    double tsat = saturatedTemperature(pressure);
    double rhog = saturatedVaporDensity(tsat);
    double vg = 1.0 / rhog;
    return internalEnergy(tsat, rhog) + pressure * vg;
}

double Properties::liquidDensity(double temperature, double pressure) const {
    return waterDensity(temperature, pressure, 1);
}

double Properties::liquidEnthalpy(double temperature, double pressure) const {
    // For subcooled liquid, calculate enthalpy properly
    double tsat = saturatedTemperature(pressure);

    if (temperature > tsat) {
        // Subcooled liquid above saturation temperature
        double h_sat = saturatedLiquidEnthalpy(pressure);
        double cp_liquid = 4200.0; // J/kg-K approximate specific heat of liquid water
        return h_sat + cp_liquid * (temperature - tsat);
    } else {
        // Use existing calculation for liquid below saturation
        double rho = liquidDensity(temperature, pressure);
        return internalEnergy(temperature, rho) + pressure / rho;
    }
}

double Properties::liquidTemperature(double pressure, double enthalpy) const {
    // Newton-Raphson iteration to find temperature
    double tsat = saturatedTemperature(pressure);
    double h_sat = saturatedLiquidEnthalpy(pressure);

    // For subcooled liquid, use a simpler approach if enthalpy is close to saturated
    if (std::abs(enthalpy - h_sat) < 10000.0) { // Within 10 kJ/kg of saturation
        // Use approximate relationship for subcooled liquid
        double cp_liquid = 4200.0; // J/kg-K for water
        double delta_h = enthalpy - h_sat;
        return tsat + delta_h / cp_liquid;
    }

    // For larger differences, use Newton-Raphson with better initial guess
    double temperature;
    if (enthalpy >= h_sat) {
        // Subcooled liquid with temperature above saturation
        double cp_liquid = 4200.0;
        temperature = tsat + (enthalpy - h_sat) / cp_liquid; // Better initial guess
    } else {
        // Subcooled liquid with temperature below saturation
        temperature = tsat - 10.0; // Initial guess below saturation
    }

    for (int iter = 0; iter < NMAXITS; ++iter) {
        double h_calc = liquidEnthalpy(temperature, pressure);
        double error = enthalpy - h_calc;

        if (std::abs(error) < ETOL * enthalpy) {
            break;
        }

        // Numerical derivative
        double delta_t = -0.001;
        double h_pert = liquidEnthalpy(temperature + delta_t, pressure);
        double dhdt = (h_pert - h_calc) / delta_t;

        double dt = -error / dhdt;
        dt = std::max(dt, -0.1 * temperature);
        dt = std::min(dt, 0.1 * temperature);

        temperature += dt;
        // Allow temperature above saturation for subcooled liquid
        // temperature = std::min(temperature, tsat); // Removed constraint
    }

    return temperature;
}

double Properties::waterDensity(double temperature, double pressure, int phase_flag) const {
    // Simplified water density calculation
    // phase_flag: 0 = vapor, 1 = liquid

    double delta = 0.1 * ETOL;
    double density;

    if (phase_flag == 0) {
        density = 1.0; // Initial guess for vapor
    } else {
        density = saturatedLiquidDensity(temperature);
    }

    for (int iter = 0; iter < NMAXITS; ++iter) {
        double p_calc = waterPressure(temperature, density);
        double error = pressure - p_calc;

        if (std::abs(error) < ETOL * pressure) {
            break;
        }

        double p_pert = waterPressure(temperature, density + delta);
        double dpdrho = (p_pert - p_calc) / delta;

        double drho = -error / dpdrho;
        double lambda = std::min(1.0, 0.5 * density / std::abs(drho));

        density += lambda * drho;
    }

    return density;
}

double Properties::waterPressure(double temperature, double density) const {
    // Simplified pressure calculation - would use full IAPWS formulation in production
    const double R = 461.51; // Specific gas constant for water [J/kg-K]

    // Simple approximation
    return density * R * temperature * (1.0 + 0.001 * density);
}

double Properties::internalEnergy(double temperature, double density) const {
    // Simplified internal energy calculation
    const double u0 = 0.0; // Reference internal energy at 0°C
    const double t0 = 273.16; // Reference temperature

    // Simple temperature dependence
    double cp_avg = 4200.0; // Approximate specific heat

    return u0 + cp_avg * (temperature - t0);
}

double Properties::distributionParameter(double void_fraction) const {
    double index = d_alpha_ * (void_fraction - alpha_0_);
    return interpolateTable(c0_table_, index, n_alpha_pts_);
}

double Properties::driftVelocity(double void_fraction) const {
    double index = d_alpha_ * (void_fraction - alpha_0_);
    return interpolateTable(vg_table_, index, n_alpha_pts_);
}

double Properties::densityFromEnthalpy(double enthalpy) const {
    double index = d_enthalpy_ * (enthalpy - h_0_);
    return interpolateTable(rho_h_table_, index, n_enthalpy_pts_);
}

double Properties::temperatureFromEnthalpy(double enthalpy) const {
    double index = d_enthalpy_ * (enthalpy - h_0_);
    return interpolateTable(t_h_table_, index, n_enthalpy_pts_);
}

double Properties::densityFromTemperature(double temperature) const {
    double index = d_temp_ * (temperature - t_0_);
    return interpolateTable(rho_t_table_, index, n_temp_pts_);
}

double Properties::enthalpyFromTemperature(double temperature) const {
    double index = d_temp_ * (temperature - t_0_);
    return interpolateTable(h_t_table_, index, n_temp_pts_);
}

double Properties::interpolateTable(const std::vector<double>& table, double index, int n_pts) const {
    int i0 = static_cast<int>(index);
    double frac = index - i0;

    if (i0 >= n_pts) {
        return table[n_pts];
    } else if (i0 < 0) {
        return table[0];
    } else {
        int i1 = std::min(i0 + 1, n_pts);
        return table[i0] + (table[i1] - table[i0]) * frac;
    }
}

} // namespace subchannel
} // namespace ants
