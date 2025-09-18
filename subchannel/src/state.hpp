#pragma once

#include "vectors.hpp"

struct State {
    Water fluid; // reference to fluid properties

    Vector1D h_l;       // liquid enthalpy
    Vector1D W_l;       // liquid mass flow rate
    Vector1D W_v;       // vapor mass flow rate
    Vector1D P;         // pressure
    Vector1D alpha;     // void fraction
    Vector1D X;         // quality
    Vector1D lhr;       // linear heat rate
    Vector1D evap;      // evaporation term [kg/m/s]

    // mixture enthalpy
    Vector1D h_m() const {
        if (W_l.size() != W_v.size() || h_l.size() != W_l.size()) {
            throw std::length_error("State::h_m(): h_l, W_l, and W_v have different sizes");
        }
        Vector1D out(h_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = h_m(i);
        }
        return out;
    }

    double h_m(size_t i) const {
        return (1 - X[i]) * h_l[i] + X[i] * fluid.h_g();
    }

    // mixture mass flow rate
    Vector1D W_m() const {
        if (W_l.size() != W_v.size()) {
            throw std::length_error("State::W_m(): W_l and W_v have different sizes");
        }
        Vector1D out(W_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = W_l[i] + W_v[i];
        }
        return out;
    }
    double W_m(size_t i) const {
        return W_l[i] + W_v[i];
    }
};
