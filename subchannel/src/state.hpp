#pragma once

#include "vectors.hpp"

struct State {
    Vector1D h;      // enthalpy
    Vector1D P;      // pressure
    Vector1D W_l;    // liquid mass flow rate
    Vector1D W_v;    // vapor mass flow rate
    Vector1D alpha;  // void fraction

    // mixture mass flow rate
    Vector1D W_m() const {
        return W_l + W_v;
    }
    double W_m(size_t i) const {
        return W_l[i] + W_v[i];
    }
};
