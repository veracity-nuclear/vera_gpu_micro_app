#pragma once

#include <vector>
#include <stdexcept>

#include "vectors.hpp"

class Water {
public:
    Water() = default;
    ~Water() = default;

    // temperature [K]
    double T(double h);
    Vector1D T(const Vector1D& h);
    double Tsat() { return 285.83 + 273.15; } // K, saturation temperature at 7 MPa

    // enthalpy [kJ/kg]
    double h(double T);
    Vector1D h(const Vector1D& T);
    double h_f() { return 1263.1; } // saturated liquid enthalpy at saturation temperature
    double h_g() { return 2773.7; } // saturated vapor enthalpy at saturation temperature
    double h_fg() { return h_g() - h_f(); } // latent heat of vaporization at saturation temperature

    // specific volume [m^3/kg]
    double v_f() { return 0.001349; } // saturated liquid specific volume at saturation temperature
    double v_g() { return 0.027756; } // saturated vapor specific volume at saturation temperature

    // density [kg/m^3]
    double rho(double h);
    Vector1D rho(const Vector1D& h);
    double rho_f() { return 1 / v_f(); } // saturated liquid density at saturation temperature
    double rho_g() { return 1 / v_g(); } // saturated vapor density at saturation temperature

    // specific heat [kJ/kg-K]
    double Cp(double h);
    Vector1D Cp(const Vector1D& h);

    // viscosity [Pa-s]
    double mu(double h);
    Vector1D mu(const Vector1D& h);

    // thermal conductivity [W/m-K]
    double k(double h);
    Vector1D k(const Vector1D& h);

    // surface tension [N/m]
    double sigma() { return 0.02; } // N/m, approximate value for water
};
