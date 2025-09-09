#pragma once

#include <vector>
#include <stdexcept>

class Water {
public:
    Water() = default;
    ~Water() = default;

    // enthalpy [kJ/kg]
    double h(double T);
    std::vector<double> h(std::vector<double> T);

    // temperature [K]
    double T(double h);
    std::vector<double> T(std::vector<double> h);

    // density [kg/m^3]
    double rho(double h);
    std::vector<double> rho(std::vector<double> h);

    // specific heat [kJ/kg-K]
    double Cp(double h);
    std::vector<double> Cp(std::vector<double> h);

    // viscosity [Pa-s]
    double mu(double h);
    std::vector<double> mu(std::vector<double> h);

    // thermal conductivity [W/m-K]
    double k(double h);
    std::vector<double> k(std::vector<double> h);
};
