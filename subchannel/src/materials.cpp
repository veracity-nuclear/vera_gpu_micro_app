#include "materials.hpp"

double Water::h(double T) const {
    // Simple linear approximation for water enthalpy
    return Cp(0) * (T - 273.15); // J/kg
}

Vector3D Water::h(const Vector3D& T) const {
    Vector3D h_values;
    Vector::resize(h_values, T.size(), T[0].size(), T[0][0].size());
    for (size_t i = 0; i < T.size(); ++i) {
        for (size_t j = 0; j < T[i].size(); ++j) {
            for (size_t k = 0; k < T[i][j].size(); ++k) {
                h_values[i][j][k] = h(T[i][j][k]);
            }
        }
    }
    return h_values;
}

double Water::T(double h) const {
    return h / 4220.0 + 273.15; // K, only true because specific heat is constant
}

Vector3D Water::T(const Vector3D& h) const {
    Vector3D T_values;
    Vector::resize(T_values, h.size(), h[0].size(), h[0][0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t j = 0; j < h[i].size(); ++j) {
            for (size_t k = 0; k < h[i][j].size(); ++k) {
                T_values[i][j][k] = T(h[i][j][k]);
            }
        }
    }
    return T_values;
}

double Water::rho(double h) const {
    // Simple approximation for water density
    return 958.0; // kg/m^3
}

Vector3D Water::rho(const Vector3D& h) const {
    Vector3D rho_values;
    Vector::resize(rho_values, h.size(), h[0].size(), h[0][0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t j = 0; j < h[i].size(); ++j) {
            for (size_t k = 0; k < h[i][j].size(); ++k) {
                rho_values[i][j][k] = rho(h[i][j][k]);
            }
        }
    }
    return rho_values;
}

double Water::Cp(double h) const {
    // Simple approximation for water specific heat capacity
    return 4220.0; // J/kg-K
}

Vector3D Water::Cp(const Vector3D& h) const {
    Vector3D Cp_values;
    Vector::resize(Cp_values, h.size(), h[0].size(), h[0][0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t j = 0; j < h[i].size(); ++j) {
            for (size_t k = 0; k < h[i][j].size(); ++k) {
                Cp_values[i][j][k] = Cp(h[i][j][k]);
            }
        }
    }
    return Cp_values;
}

double Water::mu(double h) const {
    // Simple approximation for water viscosity (value for saturated liquid at 7 MPa)
    return 0.001352; // Pa-s
}

Vector3D Water::mu(const Vector3D& h) const {
    Vector3D mu_values;
    Vector::resize(mu_values, h.size(), h[0].size(), h[0][0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t j = 0; j < h[i].size(); ++j) {
            for (size_t k = 0; k < h[i][j].size(); ++k) {
                mu_values[i][j][k] = mu(h[i][j][k]);
            }
        }
    }
    return mu_values;
}

double Water::k(double h) const {
    // Simple approximation for water thermal conductivity (approximate value for water at 250°C, 7 MPa)
    return 0.6; // W/m-K
}

Vector3D Water::k(const Vector3D& h) const {
    Vector3D k_values;
    Vector::resize(k_values, h.size(), h[0].size(), h[0][0].size());
    for (size_t i = 0; i < h.size(); ++i) {
        for (size_t j = 0; j < h[i].size(); ++j) {
            for (size_t plane = 0; plane < h[i][j].size(); ++plane) {
                k_values[i][j][plane] = k(h[i][j][plane]);
            }
        }
    }
    return k_values;
}
