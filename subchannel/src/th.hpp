#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>

#include "vectors.hpp"
#include "constants.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"
#include "linear_algebra.hpp"

namespace TH {

void planar(State& state, Vector1D mmix, Vector1D mdrift, Vector1D wbarms, Vector1D wbarvs, Vector1D whbars, Vector1D &wvms);
void FUNCV_ENTHALPY(State& state, double wmm, double wbarms, double hmm, double whbars, double qz, double wmgm,
    double wbarvs, double gam, double wmlp, double wmgp, double wmp, double hlp, double hmp, double xfp);
double FUNCV_ALPHA(State& state, size_t k, size_t ij, double wmlp, double wmgp);
double DELTAP_AXIAL(State& state, size_t k, size_t ij, double wmlp, double wmgp, double alphzp, double xfp, double gvp, double gvm, double wvms);
void solve_evaporation_term(State& state);
void solve_mixing(State& state);
void solve_surface_mass_flux(State& state);
void solve_flow_rates(State& state);
void solve_enthalpy(State& state);
void solve_void_fraction(State& state);
void solve_quality(State& state);
void solve_pressure(State& state);
double __Reynolds(double G, double D_h, double mu);
double __Prandtl(double Cp, double mu, double k);
double __Peclet(double Re, double Pr);
double __liquid_velocity(double W_l, double A_f, double alpha, double rho_l);
double __vapor_velocity(double W_v, double A_f, double alpha, double rho_g);
double __eddy_velocity(double Re, double S_ij, double D_H_i, double D_H_j, double D_rod, double G_m_i, double rho_m);
double __quality_avg(double G_m_i, double G_m_j);

} // namespace TH
