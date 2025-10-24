#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "vectors.hpp"
#include "constants.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"

namespace TH {

void solve_evaporation_term(State& state);
void solve_turbulent_mixing(State& state);
void solve_void_drift(State& state);
void solve_surface_mass_flux(State& state);
void solve_flow_rates(State& state);
void solve_enthalpy(State& state);
void solve_void_fraction(State& state);
void solve_quality(State& state);
void solve_pressure(State& state);

} // namespace TH
