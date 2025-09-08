#pragma once

#include "vectors.hpp"

struct State {
    Vector1D h;      // enthalpy
    Vector1D P;      // pressure
    Vector1D W_l;    // liquid mass flow rate
    Vector1D W_v;    // vapor mass flow rate
    Vector1D alpha;  // void fraction
};
