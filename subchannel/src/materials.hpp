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
    Vector2D T(const Vector2D& h) const;
    double Tsat() const { return 285.83 + 273.15; } // K, saturation temperature at 7 MPa

    // enthalpy [J/kg]
    double h(double T) const;
    Vector2D h(const Vector2D& T) const;
    double h_f() const { return 1263.1e3; } // saturated liquid enthalpy at saturation temperature
    double h_g() const { return 2773.7e3; } // saturated vapor enthalpy at saturation temperature
    double h_fg() const { return h_g() - h_f(); } // latent heat of vaporization at saturation temperature

    // specific volume [m^3/kg]
    double v_f() const { return 0.001349; } // saturated liquid specific volume at saturation temperature
    double v_g() const { return 0.027756; } // saturated vapor specific volume at saturation temperature
    double v_fg() const { return v_g() - v_f(); } // difference in specific volume at saturation temperature

    // density [kg/m^3]
    double rho(double h) const;
    Vector2D rho(const Vector2D& h) const;
    double rho_f() const { return 1 / v_f(); } // saturated liquid density at saturation temperature
    double rho_g() const { return 1 / v_g(); } // saturated vapor density at saturation temperature
    double rho_m(double X) const { return 1 / (X * v_g() + (1 - X) * v_f()); } // mixture density at saturation temperature

    // specific heat [J/kg-K]
    double Cp(double h) const;
    Vector2D Cp(const Vector2D& h) const;

    // viscosity [Pa-s]
    double mu(double h) const;
    Vector2D mu(const Vector2D& h) const;
    double mu_f() const { return 91.266e-6; } // saturated liquid viscosity at saturation temperature
    double mu_g() const { return 18.890e-6; } // saturated vapor viscosity at saturation temperature

    // thermal conductivity [W/m-K]
    double k(double h) const;
    Vector2D k(const Vector2D& h) const;

    // surface tension [N/m]
    double sigma() const { return 0.02; } // N/m, approximate value for water
};
