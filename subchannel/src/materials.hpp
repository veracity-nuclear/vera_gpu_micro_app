#pragma once

#include <vector>
#include <stdexcept>

#include "vectors.hpp"

class Water {
public:
    Water() = default;
    ~Water() = default;

    // temperature [K]
    double T(double h) const;
    Vector1D T(const Vector1D& h) const;
    double Tsat() const { return 285.83 + 273.15; } // K, saturation temperature at 7 MPa

    // enthalpy [kJ/kg]
    double h(double T) const;
    Vector1D h(const Vector1D& T) const;
    double h_f() const { return 1263.1; } // saturated liquid enthalpy at saturation temperature
    double h_g() const { return 2773.7; } // saturated vapor enthalpy at saturation temperature
    double h_fg() const { return h_g() - h_f(); } // latent heat of vaporization at saturation temperature

    // specific volume [m^3/kg]
    double v_f() const { return 0.001349; } // saturated liquid specific volume at saturation temperature
    double v_g() const { return 0.027756; } // saturated vapor specific volume at saturation temperature

    // density [kg/m^3]
    double rho(double h) const;
    Vector1D rho(const Vector1D& h) const;
    double rho_f() const { return 1 / v_f(); } // saturated liquid density at saturation temperature
    double rho_g() const { return 1 / v_g(); } // saturated vapor density at saturation temperature

    // specific heat [kJ/kg-K]
    double Cp(double h) const;
    Vector1D Cp(const Vector1D& h) const;

    // viscosity [Pa-s]
    double mu(double h) const;
    Vector1D mu(const Vector1D& h) const;
    double mu_f() const { return 91.266e-6; } // saturated liquid viscosity at saturation temperature
    double mu_g() const { return 18.890e-6; } // saturated vapor viscosity at saturation temperature

    // thermal conductivity [W/m-K]
    double k(double h) const;
    Vector1D k(const Vector1D& h) const;

    // surface tension [N/m]
    double sigma() const { return 0.02; } // N/m, approximate value for water
};
