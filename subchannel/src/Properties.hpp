#ifndef ANTS_SUBCHANNEL_PROPERTIES_HPP
#define ANTS_SUBCHANNEL_PROPERTIES_HPP

#include "Constants.hpp"
#include <array>
#include <vector>

namespace ants {
namespace subchannel {

/**
 * @brief Thermophysical properties and water property correlations
 * 
 * This class provides water and steam properties along with
 * drift-flux correlations for two-phase flow calculations.
 */
class Properties {
public:
    Properties();
    ~Properties() = default;

    // Steam table initialization
    void initialize();

    // Water property correlations
    double saturatedTemperature(double pressure) const;
    double saturatedPressure(double temperature) const;
    double saturatedPressureDerivative(double temperature) const;
    double saturatedLiquidDensity(double temperature) const;
    double saturatedVaporDensity(double temperature) const;
    double saturatedLiquidEnthalpy(double pressure) const;
    double saturatedVaporEnthalpy(double pressure) const;
    
    // General water properties
    double liquidDensity(double temperature, double pressure) const;
    double liquidEnthalpy(double temperature, double pressure) const;
    double liquidTemperature(double pressure, double enthalpy) const;
    double vaporDensity(double temperature, double pressure) const;
    double vaporEnthalpy(double temperature, double pressure) const;
    
    // Internal energy and pressure
    double internalEnergy(double temperature, double density) const;
    double waterPressure(double temperature, double density) const;
    double waterDensity(double temperature, double pressure, int phase_flag) const;

    // Drift-flux correlations (Chexal-Lellouche)
    double distributionParameter(double void_fraction) const;
    double driftVelocity(double void_fraction) const;

    // Table interpolation functions
    double densityFromEnthalpy(double enthalpy) const;
    double temperatureFromEnthalpy(double enthalpy) const;
    double densityFromTemperature(double temperature) const;
    double enthalpyFromTemperature(double temperature) const;

    // Getter functions for key properties
    double getSaturationTemperature() const { return t_sat_; }
    double getLiquidDensity() const { return rho_liquid_; }
    double getVaporDensity() const { return rho_vapor_; }
    double getLiquidEnthalpy() const { return h_liquid_; }
    double getVaporEnthalpy() const { return h_vapor_; }
    double getEnthalpyOfVaporization() const { return h_fg_; }

    // Set operating conditions
    void setOperatingPressure(double pressure);

private:
    // Steam table data
    int n_alpha_pts_;
    double d_alpha_;
    double alpha_0_;
    std::vector<double> c0_table_;
    std::vector<double> vg_table_;
    
    int n_enthalpy_pts_;
    double d_enthalpy_;
    double h_0_;
    std::vector<double> rho_h_table_;
    std::vector<double> t_h_table_;
    
    int n_temp_pts_;
    double d_temp_;
    double t_0_;
    std::vector<double> rho_t_table_;
    std::vector<double> h_t_table_;

    // Operating conditions
    double operating_pressure_;
    double t_sat_;
    double rho_liquid_;
    double rho_vapor_;
    double h_liquid_;
    double h_vapor_;
    double h_fg_;
    
    // Thermophysical property data
    static constexpr int NPTS = 32;
    static constexpr std::array<double, NPTS> viscosity_table_ = {
        0.95933e-4, 0.94445e-4, 0.93038e-4, 0.91703e-4, 0.90429e-4, 0.89211e-4, 0.88042e-4, 0.86917e-4,
        0.85831e-4, 0.84781e-4, 0.83761e-4, 0.82770e-4, 0.81805e-4, 0.80863e-4, 0.79941e-4, 0.79038e-4,
        0.78152e-4, 0.77281e-4, 0.76423e-4, 0.75577e-4, 0.74741e-4, 0.73913e-4, 0.73094e-4, 0.72280e-4,
        0.71471e-4, 0.70665e-4, 0.69861e-4, 0.69056e-4, 0.68250e-4, 0.67440e-4, 0.66624e-4, 0.65810e-4
    };
    
    static constexpr std::array<double, NPTS> surface_tension_table_ = {
        0.20404e-1, 0.19529e-1, 0.18693e-1, 0.17891e-1, 0.17122e-1, 0.16382e-1, 0.15670e-1, 0.14984e-1,
        0.14322e-1, 0.13682e-1, 0.13064e-1, 0.12467e-1, 0.11888e-1, 0.11328e-1, 0.10785e-1, 0.10258e-1,
        0.97474e-2, 0.92520e-2, 0.87712e-2, 0.83045e-2, 0.78515e-2, 0.74116e-2, 0.69844e-2, 0.65697e-2,
        0.61670e-2, 0.57760e-2, 0.53966e-2, 0.50283e-2, 0.46711e-2, 0.43247e-2, 0.39890e-2, 0.36638e-2
    };

    // Helper functions
    double interpolateTable(const std::vector<double>& table, double index, int n_pts) const;
    void buildSteamTables();
    
    // Newton-Raphson solver for property inversions
    double solveForTemperature(double target_pressure, double initial_guess) const;
    double solveForDensity(double temperature, double pressure, int phase_flag) const;
};

} // namespace subchannel
} // namespace ants

#endif // ANTS_SUBCHANNEL_PROPERTIES_HPP
