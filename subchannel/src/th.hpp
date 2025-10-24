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
std::pair<size_t, size_t> __get_neighbor_ij(size_t i, size_t j, size_t ns, size_t nx, size_t ny);
double __Reynolds(double G, double D_h, double mu);
double __Prandtl(double Cp, double mu, double k);
double __Peclet(double Re, double Pr);
double __liquid_velocity(double W_l, double A_f, double alpha, double rho_l);
double __vapor_velocity(double W_v, double A_f, double alpha, double rho_g);
double __eddy_velocity(double Re, double S_ij, double D_H_i, double D_H_j, double D_rod, double G_m_i, double X, double rho_m);
double __two_phase_multiplier(double X_bar, double Re);
double __quality_avg(double G_m_i, double G_m_j);

} // namespace TH
