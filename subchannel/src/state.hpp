#pragma once

#include "vectors.hpp"

struct State {
    Water fluid; // reference to fluid properties

    Vector3D h_l;       // liquid enthalpy
    Vector3D W_l;       // liquid mass flow rate
    Vector3D W_v;       // vapor mass flow rate
    Vector3D P;         // pressure
    Vector3D alpha;     // void fraction
    Vector3D X;         // quality
    Vector3D lhr;       // linear heat rate
    Vector3D evap;      // evaporation term [kg/m/s]

    // mixture enthalpy
    Vector3D h_m() const {
        if (W_l.size() != W_v.size() || h_l.size() != W_l.size()) {
            throw std::length_error("State::h_m(): h_l, W_l, and W_v have different sizes");
        }
        Vector3D out(h_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            for (std::size_t j = 0; j < out[i].size(); ++j) {
                for (std::size_t k = 0; k < out[i][j].size(); ++k) {
                    double X_ijk = X[i][j][k];
                    out[i][j][k] = (1 - X_ijk) * h_l[i][j][k] + X_ijk * fluid.h_g();
                }
            }
        }
        return out;
    }

    double h_m(size_t i, size_t j, size_t k) const {
        return (1 - X[i][j][k]) * h_l[i][j][k] + X[i][j][k] * fluid.h_g();
    }

    // mixture mass flow rate
    Vector3D W_m() const {
        if (W_l.size() != W_v.size()) {
            throw std::length_error("State::W_m(): W_l and W_v have different sizes");
        }
        Vector3D out(W_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            for (std::size_t j = 0; j < out[i].size(); ++j) {
                for (std::size_t k = 0; k < out[i][j].size(); ++k) {
                    out[i][j][k] = W_l[i][j][k] + W_v[i][j][k];
                }
            }
        }
        return out;
    }
    double W_m(size_t i, size_t j, size_t k) const {
        return W_l[i][j][k] + W_v[i][j][k];
    }
};
