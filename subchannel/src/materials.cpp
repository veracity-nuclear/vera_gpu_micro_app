#include "materials.hpp"

double Water::h(double T) const {
    // Simple linear approximation for water enthalpy
    return Cp(0) * (T - 273.15); // J/kg
}

Vector2D Water::h(const Vector2D& T) const {
    Vector2D h_values;
    Vector::resize(h_values, T.size(), T[0].size());
    for (size_t i = 0; i < T.size(); ++i) {
        for (size_t k = 0; k < T[i].size(); ++k) {
            h_values[i][k] = h(T[i][k]);
        }
    }
    return h_values;
}

double Water::T(double h) const {
    return h / 4220.0 + 273.15; // K, only true because specific heat is constant
}

Vector2D Water::T(const Vector2D& h) const {
    Vector2D T_values;
    Vector::resize(T_values, h.size(), h[0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t k = 0; k < h[i].size(); ++k) {
            T_values[i][k] = T(h[i][k]);
        }
    }
    return T_values;
}

double Water::rho(double h) const {
    // Simple approximation for water density
    return 958.0; // kg/m^3
}

Vector2D Water::rho(const Vector2D& h) const {
    Vector2D rho_values;
    Vector::resize(rho_values, h.size(), h[0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t k = 0; k < h[i].size(); ++k) {
            rho_values[i][k] = rho(h[i][k]);
        }
    }
    return rho_values;
}

double Water::Cp(double h) const {
    // Simple approximation for water specific heat capacity
    return 4220.0; // J/kg-K
}

Vector2D Water::Cp(const Vector2D& h) const {
    Vector2D Cp_values;
    Vector::resize(Cp_values, h.size(), h[0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t k = 0; k < h[i].size(); ++k) {
            Cp_values[i][k] = Cp(h[i][k]);
        }
    }
    return Cp_values;
}

double Water::mu(double h) const {
    // Simple approximation for water viscosity (value for saturated liquid at 7 MPa)
    return 0.001352; // Pa-s
}

Vector2D Water::mu(const Vector2D& h) const {
    Vector2D mu_values;
    Vector::resize(mu_values, h.size(), h[0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t k = 0; k < h[i].size(); ++k) {
            mu_values[i][k] = mu(h[i][k]);
        }
    }
    return mu_values;
}

double Water::k(double h) const {
    // Simple approximation for water thermal conductivity (approximate value for water at 250°C, 7 MPa)
    return 0.6; // W/m-K
}

Vector2D Water::k(const Vector2D& h) const {
    Vector2D k_values;
    Vector::resize(k_values, h.size(), h[0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t plane = 0; plane < h[i].size(); ++plane) {
            k_values[i][plane] = k(h[i][plane]);
        }
    }
    return k_values;
}
