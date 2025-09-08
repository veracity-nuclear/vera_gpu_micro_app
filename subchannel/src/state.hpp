#pragma once

#include "vectors.hpp"

struct State {
    Vector1D h;         // enthalpy
    Vector1D P;         // pressure
    Vector1D W_l;       // liquid mass flow rate
    Vector1D W_v;       // vapor mass flow rate
    Vector1D alpha;     // void fraction
    Vector1D lhr;       // linear heat rate

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
