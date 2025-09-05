#pragma once

#include <vector>
#include <stdexcept>

class Water {
public:
    Water() = default;
    ~Water() = default;

    // temperature [K]
    double T(double h);
    std::vector<double> T(std::vector<double> h);
    double Tsat() { return 285.83 + 273.15; } // K, saturation temperature at 7 MPa

    // enthalpy [kJ/kg]
    double h(double T);
    std::vector<double> h(std::vector<double> T);
    double h_f() { return 1263.1; } // saturated liquid enthalpy at saturation temperature
    double h_g() { return 2773.7; } // saturated vapor enthalpy at saturation temperature
    double h_fg() { return h_g() - h_f(); } // latent heat of vaporization at saturation temperature

    // specific volume [m^3/kg]
    double v_f() { return 0.001349; } // saturated liquid specific volume at saturation temperature
    double v_g() { return 0.027756; } // saturated vapor specific volume at saturation temperature

    // density [kg/m^3]
    double rho(double h);
    std::vector<double> rho(std::vector<double> h);
    double rho_f() { return 1 / v_f(); } // saturated liquid density at saturation temperature
    double rho_g() { return 1 / v_g(); } // saturated vapor density at saturation temperature

    // specific heat [kJ/kg-K]
    double Cp(double h);
    std::vector<double> Cp(std::vector<double> h);

    // viscosity [Pa-s]
    double mu(double h);
    std::vector<double> mu(std::vector<double> h);

    // thermal conductivity [W/m-K]
    double k(double h);
    std::vector<double> k(std::vector<double> h);

    // surface tension [N/m]
    double sigma() { return 0.02; } // N/m, approximate value for water
};
