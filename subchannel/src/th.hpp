#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"
#include "linear_algebra.hpp"

namespace TH {

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void planar(State<ExecutionSpace>& state);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void accumulate_surface_sources(State<ExecutionSpace>& state);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_evaporation_term(State<ExecutionSpace>& state);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_mixing(State<ExecutionSpace>& state);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_surface_mass_flux(State<ExecutionSpace>& state);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_flow_rates(
    size_t ij, size_t k, size_t k_node, double A_f, double dz,
    const typename State<ExecutionSpace>::View2D& evap,
    const typename State<ExecutionSpace>::View1D& SS_l,
    const typename State<ExecutionSpace>::View1D& SS_v,
    typename State<ExecutionSpace>::View2D& W_l,
    typename State<ExecutionSpace>::View2D& W_v
);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_enthalpy(
    size_t ij, size_t k, size_t k_node, double dz, double gap_width, double h_g,
    const typename State<ExecutionSpace>::View2D& W_l,
    const typename State<ExecutionSpace>::View2D& W_v,
    const typename State<ExecutionSpace>::View2D& lhr,
    const typename State<ExecutionSpace>::View1D& SS_m,
    typename State<ExecutionSpace>::View2D& h_l
);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_void_fraction(
    size_t ij, size_t k, size_t k_node, double A_f, double D_h, double rho_f, double rho_g,
    double h_f, double h_fg, double mu_v, double sigma, size_t max_inner_iter,
    const typename State<ExecutionSpace>::View2D& P,
    const typename State<ExecutionSpace>::View2D& W_l,
    const typename State<ExecutionSpace>::View2D& W_v,
    const typename State<ExecutionSpace>::View2D& h_l,
    const typename State<ExecutionSpace>::View2D& X,
    const typename State<ExecutionSpace>::View2D& rho,
    const typename State<ExecutionSpace>::View2D& mu,
    typename State<ExecutionSpace>::View2D& alpha
);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_quality(
    size_t ij, size_t k, size_t k_node, double A_f,
    const typename State<ExecutionSpace>::View2D& W_l,
    const typename State<ExecutionSpace>::View2D& W_v,
    typename State<ExecutionSpace>::View2D& X
);

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_pressure(
    size_t ij, size_t k, size_t k_node, double A_f, double D_h, double dz,
    double rho_f, double rho_g, double mu_f, double mu_g,
    const typename State<ExecutionSpace>::View2D& W_l,
    const typename State<ExecutionSpace>::View2D& W_v,
    const typename State<ExecutionSpace>::View2D& h_l,
    const typename State<ExecutionSpace>::View2D& X,
    const typename State<ExecutionSpace>::View2D& alpha,
    const typename State<ExecutionSpace>::View1D& CF_SS,
    const typename State<ExecutionSpace>::View1D& TM_SS,
    const typename State<ExecutionSpace>::View1D& VD_SS,
    const typename State<ExecutionSpace>::View2D& rho,
    const typename State<ExecutionSpace>::View2D& mu,
    typename State<ExecutionSpace>::View2D& P
);

double __Reynolds(double G, double D_h, double mu);
double __Prandtl(double Cp, double mu, double k);
double __Peclet(double Re, double Pr);
double __liquid_velocity(double W_l, double A_f, double alpha, double rho_l);
double __vapor_velocity(double W_v, double A_f, double alpha, double rho_g);
double __eddy_velocity(double Re, double S_ij, double D_H_i, double D_H_j, double D_rod, double G_m_i, double rho_m);
double __quality_avg(double G_m_i, double G_m_j);

} // namespace TH
