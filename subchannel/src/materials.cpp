#include "materials.hpp"

double Water::h(double T) {
    // Simple linear approximation for water enthalpy
    return Cp(0) * (T - 273.15); // kJ/kg
}

std::vector<double> Water::h(std::vector<double> T) {
    std::vector<double> h_values;
    for (double T_i : T) {
        h_values.push_back(h(T_i));
    }
    return h_values;
}

double Water::T(double h) {
    return h / 4.22 + 273.15; // K, only true because specific heat is constant
}

std::vector<double> Water::T(std::vector<double> h) {
    std::vector<double> T_values;
    for (double h_i : h) {
        T_values.push_back(T(h_i));
    }
    return T_values;
}

double Water::rho(double h) {
    // Simple approximation for water density
    return 958.0; // kg/m^3
}

std::vector<double> Water::rho(std::vector<double> h) {
    std::vector<double> rho_values;
    for (double h_i : h) {
        rho_values.push_back(rho(h_i));
    }
    return rho_values;
}

double Water::Cp(double h) {
    // Simple approximation for water specific heat capacity
    return 4.22; // kJ/kg-K
}

std::vector<double> Water::Cp(std::vector<double> h) {
    std::vector<double> Cp_values;
    for (double h_i : h) {
        Cp_values.push_back(Cp(h_i));
    }
    return Cp_values;
}

double Water::mu(double h) {
    // Simple approximation for water viscosity (value for saturated liquid at 7 MPa)
    return 0.001352; // Pa-s
}

std::vector<double> Water::mu(std::vector<double> h) {
    std::vector<double> mu_values;
    for (double h_i : h) {
        mu_values.push_back(mu(h_i));
    }
    return mu_values;
}

double Water::k(double h) {
    // Simple approximation for water thermal conductivity (approximate value for water at 250°C, 7 MPa)
    return 0.6; // W/m-K
}

std::vector<double> Water::k(std::vector<double> h) {
    std::vector<double> k_values;
    for (double h_i : h) {
        k_values.push_back(k(h_i));
    }
    return k_values;
}
