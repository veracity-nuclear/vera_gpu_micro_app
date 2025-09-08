#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "vectors.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"

namespace TH {

void solve_enthalpy(State& state, double lhr, const Geometry& geom);
void solve_pressure(State& state, const Geometry& geom, const Water& fluid);

} // namespace TH
