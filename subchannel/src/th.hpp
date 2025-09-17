#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

#include "vectors.hpp"
#include "constants.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"

namespace TH {

void solve_evaporation_term(State& state, const Geometry& geom, const Water& fluid);
void solve_turbulent_mixing(State& state, const Geometry& geom, const Water& fluid);
void solve_void_drift(State& state, const Geometry& geom, const Water& fluid);
void solve_surface_mass_flux(State& state, const Geometry& geom, const Water& fluid);
void solve_flow_rates(State& state, const Geometry& geom, const Water& fluid);
void solve_enthalpy(State& state, const Geometry& geom, const Water& fluid);
void solve_void_fraction(State& state, const Geometry& geom, const Water& fluid);
void solve_quality(State& state, const Geometry& geom, const Water& fluid);
void solve_pressure(State& state, const Geometry& geom, const Water& fluid);

} // namespace TH
