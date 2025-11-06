#pragma once

#include "vectors.hpp"

struct State {
    size_t surface_plane = 0;       // current surface axial plane being solved
    size_t node_plane = 0;          // current node axial plane being solved
    std::shared_ptr<Water> fluid;   // reference to fluid properties
    std::shared_ptr<Geometry> geom; // reference to geometry

    Vector2D h_l;       // liquid enthalpy
    Vector2D W_l;       // liquid mass flow rate
    Vector2D W_v;       // vapor mass flow rate
    Vector2D P;         // pressure
    Vector2D alpha;     // void fraction
    Vector2D X;         // quality
    Vector2D lhr;       // linear heat rate
    Vector2D evap;      // evaporation term [kg/m/s]

    Vector3D G_l_tm;    // turbulent mixing liquid mass transfer [kg/m^2/s]
    Vector3D G_v_tm;    // turbulent mixing vapor mass transfer [kg/m^2/s]
    Vector3D Q_m_tm;    // turbulent mixing energy transfer [W/m^2]
    Vector3D M_m_tm;    // turbulent mixing momentum transfer [Pa]

    Vector3D G_l_vd;    // void drift liquid mass transfer [kg/m^2/s]
    Vector3D G_v_vd;    // void drift vapor mass transfer [kg/m^2/s]
    Vector3D Q_m_vd;    // void drift energy transfer [W/m^2]
    Vector3D M_m_vd;    // void drift momentum transfer [Pa]

    // mixture enthalpy
    Vector2D h_m() const {
        if (W_l.size() != W_v.size() || h_l.size() != W_l.size()) {
            throw std::length_error("State::h_m(): h_l, W_l, and W_v have different sizes");
        }
        Vector2D out(h_l.size());
        for (size_t nc = 0; nc < out.size(); ++nc) {
            for (size_t k = 0; k < out[nc].size(); ++k) {
                double X_ijk = X[nc][k];
                out[nc][k] = h_m(nc, k);
            }
        }
        return out;
    }

    double h_m(size_t nc, size_t k) const {
        return (1 - X[nc][k]) * h_l[nc][k] + X[nc][k] * fluid->h_g();
    }

    // mixture mass flow rate
    Vector2D W_m() const {
        if (W_l.size() != W_v.size()) {
            throw std::length_error("State::W_m(): W_l and W_v have different sizes");
        }
        Vector2D out(W_l.size());
        for (size_t nc = 0; nc < out.size(); ++nc) {
            for (size_t k = 0; k < out[nc].size(); ++k) {
                out[nc][k] = W_m(nc, k);
            }
        }
        return out;
    }

    double W_m(size_t nc, size_t k) const {
        return W_l[nc][k] + W_v[nc][k];
    }

    // mixture velocity
    Vector2D V_m() const {
        Vector2D out(W_l.size());
        for (size_t nc = 0; nc < out.size(); ++nc) {
            for (size_t k = 0; k < out[nc].size(); ++k) {
                out[nc][k] = V_m(nc, k);
            }
        }
        return out;
    }

    double V_m(size_t nc, size_t k) const {
        Vector2D rho = fluid->rho(h_l);
        double A_f = geom->flow_area();
        double v_m;
        if (alpha[nc][k] < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha[nc][k] > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X[nc][k]) * (1.0 - X[nc][k]) / ((1.0 - alpha[nc][k]) * rho[nc][k]) + X[nc][k] * X[nc][k] / (alpha[nc][k] * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m * W_m(nc, k) / A_f; // mixture velocity, Eq. 15 from ANTS Theory
    }

    // mixture specific volume
    Vector2D nu_m() const {
        Vector2D out(W_l.size());
        for (size_t nc = 0; nc < out.size(); ++nc) {
            for (size_t k = 0; k < out[nc].size(); ++k) {
                out[nc][k] = nu_m(nc, k);
            }
        }
        return out;
    }

    double nu_m(size_t nc, size_t k) const {
        Vector2D rho = fluid->rho(h_l);
        double v_m;
        if (alpha[nc][k] < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha[nc][k] > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X[nc][k]) * (1.0 - X[nc][k]) / ((1.0 - alpha[nc][k]) * rho[nc][k]) + X[nc][k] * X[nc][k] / (alpha[nc][k] * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m; // mixture specific volume
    }
};
