#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <utility>
#include <type_traits>
#include <Kokkos_Core.hpp>

#include "constants.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "state.hpp"
#include "linear_algebra.hpp"

namespace TH {

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
void solve_surface_mass_flux(State<ExecutionSpace>& state);

KOKKOS_INLINE_FUNCTION
double __Reynolds(double G, double D_h, double mu) {
    return G * D_h / mu;
}

KOKKOS_INLINE_FUNCTION
double __Prandtl(double Cp, double mu, double k) {
    return Cp * mu / k;
}

KOKKOS_INLINE_FUNCTION
double __liquid_velocity(double W_l, double A_f, double alpha, double rho_l) {
    if (alpha < 1.0) {
        return W_l / (A_f * (1.0 - alpha) * rho_l);
    }
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
double __vapor_velocity(double W_v, double A_f, double alpha, double rho_g) {
    if (alpha > 0.0) {
        return W_v / (A_f * alpha * rho_g);
    }
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
double __Peclet(double Re, double Pr) {
    return Re * Pr;
}

KOKKOS_INLINE_FUNCTION
double __eddy_velocity(double Re, double S_ij, double D_H_i, double D_H_j, double D_rod, double G_m_i, double rho_m) {
    double lambda = 0.0058 * (S_ij / D_rod); // Eq. 46 from ANTS Theory
    return 0.5 * lambda * pow(Re, -0.1) * (1.0 + pow(D_H_j / D_H_i, 1.5)) * D_H_i / D_rod * G_m_i / rho_m; // Eq. 45 from ANTS Theory
}

KOKKOS_INLINE_FUNCTION
double __quality_avg(double G_m_i, double G_m_j) {
    double K_M = 1.4; // constant from ANTS Theory, referenced from Lahey and Moody (1977)
    return K_M * (G_m_i - G_m_j) / (G_m_i + G_m_j); // Eq. 49 from ANTS Theory
}

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void solve_flow_rates(
    size_t ij, size_t k, size_t k_node, double A_f, double dz,
    typename State<ExecutionSpace>::View2D evap,
    typename State<ExecutionSpace>::View1D SS_l,
    typename State<ExecutionSpace>::View1D SS_v,
    typename State<ExecutionSpace>::View2D W_l,
    typename State<ExecutionSpace>::View2D W_v
) {
    // Update liquid flow rate (Eq. 61 from ANTS Theory)
    W_l(ij, k) = W_l(ij, k-1) - dz * (evap(ij, k_node) + SS_l(ij));
    W_l(ij, k) = (W_l(ij, k) > 0.0) ? W_l(ij, k) : 1e-8; // prevent negative (making this 0.0 makes enthalpy nan)

    // Update vapor flow rate (Eq. 62 from ANTS Theory)
    W_v(ij, k) = W_v(ij, k-1) + dz * (evap(ij, k_node) - SS_v(ij));
    W_v(ij, k) = (W_v(ij, k) > 0.0) ? W_v(ij, k) : 1e-8; // prevent negative (making this 0.0 makes enthalpy nan)
}

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void solve_enthalpy(
    size_t ij, size_t k, size_t k_node, double dz, double gap_width, double h_g,
    typename State<ExecutionSpace>::View2D W_l,
    typename State<ExecutionSpace>::View2D W_v,
    typename State<ExecutionSpace>::View2D lhr,
    typename State<ExecutionSpace>::View1D SS_m,
    typename State<ExecutionSpace>::View2D h_l
) {
    // Eq. 63 from ANTS Theory
    h_l(ij, k) = (
        (W_v(ij, k-1) - W_v(ij, k)) * h_g
        + W_l(ij, k-1) * h_l(ij, k-1) + dz * lhr(ij, k_node)
        - dz * SS_m(ij)
    ) / W_l(ij, k);
}

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void solve_void_fraction(
    size_t ij, size_t k, size_t k_node, double A_f, double D_h, double rho_f, double rho_g,
    double h_f, double h_fg, double mu_v, double sigma, size_t max_inner_iter,
    Water fluid,
    typename State<ExecutionSpace>::View2D P,
    typename State<ExecutionSpace>::View2D W_l,
    typename State<ExecutionSpace>::View2D W_v,
    typename State<ExecutionSpace>::View2D h_l,
    typename State<ExecutionSpace>::View2D X,
    typename State<ExecutionSpace>::View2D alpha
) {
    const double tol = 1e-8;
    const double eps = 1e-12; // small number to prevent division by zero

    // based on the Chexal-Lellouche drift flux model
    double pressure = P(0, 0); // assuming constant pressure for simplicity

    double Gv = W_v(ij, k) / A_f; // vapor mass flux
    double Gl = W_l(ij, k) / A_f; // liquid mass flux

    if (Gv < eps) {
        alpha(ij, k) = 0.0;
        return;
    }

    double h_v = h_f + X(ij, k) * h_fg;
    double rho_l = fluid.rho(h_l(ij, k));
    double mu_l = fluid.mu(h_l(ij, k));

    double Re_g = __Reynolds(W_v(ij, k) / A_f, D_h, mu_v); // local vapor Reynolds number
    double Re_f = __Reynolds(W_l(ij, k) / A_f, D_h, mu_l); // local liquid Reynolds number
    double Re = (Re_g > Re_f) ? Re_g : Re_f;
    double A1 = 1 / (1 + Kokkos::exp(-Re / 60000));
    double B1 = (0.8 < A1) ? 0.8 : A1; // from Zuber correlation
    double B2 = 1.41;

    // Inline bisection method - cannot use lambda functions inside KOKKOS_LAMBDA
    double a = 0.0;
    double b = 1.0;

    // Helper lambda-like evaluation using direct computation
    auto evaluate_f = [&](double alpha_val) {
        // calculate distribution parameter, C_0
        double C1 = 4.0 * P_crit * P_crit / (pressure * (P_crit - pressure)); // Eq. 24 from ANTS Theory
        double L = (1.0 - Kokkos::exp(-C1 * alpha_val)) / (1.0 - Kokkos::exp(-C1)); // Eq. 23 from ANTS Theory
        double K0 = B1 + (1 - B1) * Kokkos::pow(rho_g / rho_f, 0.25); // Eq. 25 from ANTS Theory
        double r = (1 + 1.57 * (rho_g / rho_f)) / (1 - B1); // Eq. 26 from ANTS Theory
        double C0 = L / (K0 + (1 - K0) * Kokkos::pow(alpha_val, r)); // Eq. 22 from ANTS Theory

        // calculate drift velocity, V_gj
        double Vgj0 = B2 * Kokkos::pow(((rho_f - rho_g) * g * sigma) / (rho_f * rho_f), 0.25); // Eq. 28 from ANTS Theory
        double Vgj = Vgj0 * Kokkos::pow(1.0 - alpha_val, B1); // Eq. 27 from ANTS Theory

        return (alpha_val * C0 - 1.0) * Gv + alpha_val * C0 * (rho_g / rho_l) * Gl + alpha_val * rho_g * Vgj;
    };

    // Bisection implementation
    double fa = evaluate_f(a);
    double fb = evaluate_f(b);

    if (Kokkos::fabs(fa) < tol) {
        alpha(ij, k) = a;
        return;
    }
    if (Kokkos::fabs(fb) < tol) {
        alpha(ij, k) = b;
        return;
    }

    // Root must be bracketed for bisection
    if (fa * fb > 0) {
        // If not bracketed, use safer default
        alpha(ij, k) = 0.0;
        return;
    }

    for (int i = 0; i < (int)max_inner_iter; i++) {
        double c = 0.5 * (a + b);
        double fc = evaluate_f(c);

        if (Kokkos::fabs(fc) < tol || (b - a) < tol) {
            alpha(ij, k) = c;
            return;
        }

        if (fa * fc < 0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    alpha(ij, k) = 0.5 * (a + b);
}

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void solve_quality(
    size_t ij, size_t k, size_t k_node, double A_f,
    typename State<ExecutionSpace>::View2D W_l,
    typename State<ExecutionSpace>::View2D W_v,
    typename State<ExecutionSpace>::View2D X
) {
    double G_v = W_v(ij, k) / A_f; // vapor mass flux (Eq. 8 from ANTS Theory)
    double G_l = W_l(ij, k) / A_f; // liquid mass flux (Eq. 9 from ANTS Theory)
    X(ij, k) = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory
}

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void solve_pressure(
    size_t ij, size_t k, size_t k_node, double A_f, double D_h, double dz,
    double rho_f, double rho_g, double mu_f, double mu_g,
    Water fluid,
    typename State<ExecutionSpace>::View2D W_l,
    typename State<ExecutionSpace>::View2D W_v,
    typename State<ExecutionSpace>::View2D h_l,
    typename State<ExecutionSpace>::View2D X,
    typename State<ExecutionSpace>::View2D alpha,
    typename State<ExecutionSpace>::View1D CF_SS,
    typename State<ExecutionSpace>::View1D TM_SS,
    typename State<ExecutionSpace>::View1D VD_SS,
    typename State<ExecutionSpace>::View2D P
) {
    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    // mass flux (liq. only)
    double G_l = W_l(ij, k) / A_f;

    // mass flux (mixture)
    double G = (W_l(ij, k) + W_v(ij, k)) / A_f;

    // ----- two-phase acceleration pressure drop -----
    double nu_m_k, nu_m_km1;
    if (alpha(ij, k) < 1e-6) {
        nu_m_k = 1.0 / rho_f;
    } else if (alpha(ij, k) > 1.0 - 1e-6) {
        nu_m_k = 1.0 / rho_g;
    } else {
        nu_m_k = (1.0 - X(ij, k)) * (1.0 - X(ij, k)) / ((1.0 - alpha(ij, k)) * fluid.rho(h_l(ij, k))) +
                    X(ij, k) * X(ij, k) / (alpha(ij, k) * rho_g);
    }

    if (alpha(ij, k-1) < 1e-6) {
        nu_m_km1 = 1.0 / rho_f;
    } else if (alpha(ij, k-1) > 1.0 - 1e-6) {
        nu_m_km1 = 1.0 / rho_g;
    } else {
        nu_m_km1 = (1.0 - X(ij, k-1)) * (1.0 - X(ij, k-1)) / ((1.0 - alpha(ij, k-1)) * fluid.rho(h_l(ij, k-1))) +
                    X(ij, k-1) * X(ij, k-1) / (alpha(ij, k-1) * rho_g);
    }

    double dP_accel = G * G * (nu_m_k - nu_m_km1);

    // ----- two-phase frictional pressure drop -----
    // Reynolds number (liq. only)
    double Re = G_l * D_h / fluid.mu(h_l(ij, k));

    // frictional pressure drop from wall shear
    double f = a_1 * Kokkos::pow(Re, n);
    double K = f * dz / D_h;
    double gamma = Kokkos::pow(rho_f / rho_g, 0.5) * Kokkos::pow(mu_g / mu_f, 0.2);

    // parameter b for two-phase multiplier (Chisholm)
    double b;
    if (gamma <= 9.5) {
        b = 55.0 / Kokkos::sqrt(G);
    } else if (gamma < 28) {
        b = 520.0 / (gamma * Kokkos::sqrt(G));
    } else {
        b = 15000.0 / (gamma * gamma * Kokkos::sqrt(G));
    }

    // two-phase multiplier for wall shear (Chisholm)
    double phi2_ch = 1.0 + (gamma * gamma - 1.0) * (b * Kokkos::pow(X(ij, k), 0.9) * Kokkos::pow((1.0 - X(ij, k)), 0.9) + Kokkos::pow(X(ij, k), 1.8));

    // two-phase wall shear pressure drop
    double dP_wall_shear = K * G * G / (2.0 * rho_f) * phi2_ch;

    // form loss coefficient (no form losses in this simple model)
    double K_loss = 0.0;

    // two-phase multiplier for form losses (homogeneous), Eq. 35 from ANTS Theory
    double phi2_hom = 1.0 + X(ij, k) * (rho_f / rho_g - 1.0);

    // two-phase geometry form loss pressure drop, Eq. 36 from ANTS Theory
    double dP_form = K_loss * G * G / (2.0 * rho_f) * phi2_hom;

    // two-phase frictional pressure drop, Eq. 36 from ANTS Theory
    double dP_tpfric = dP_wall_shear + dP_form;

    // ----- two-phase gravitational pressure drop -----
    double dP_grav = fluid.rho(h_l(ij, k)) * g * dz;

    // ----- momentum exchange -----
    double dP_CF = dz * CF_SS(ij);
    double dP_TM = dz * TM_SS(ij);
    double dP_VD = dz * VD_SS(ij);
    double dP_momexch = dP_CF + dP_TM + dP_VD;

    // ----- total pressure drop -----
    double dP_total = dP_accel + dP_tpfric + dP_grav + dP_momexch;
    P(ij, k) = P(ij, k-1) - dP_total;
}

template <typename ExecutionSpace>
struct ANTSFunctor {
    using StateType = State<ExecutionSpace>;
    using View1D    = typename StateType::View1D; // double*
    using View2D    = typename StateType::View2D; // double**

    // Geometry view types deduced from Geometry<ExecutionSpace>
    using SurfaceView = decltype(std::declval<Geometry<ExecutionSpace>>().surface_view());
    using IndexView1D = decltype(std::declval<Geometry<ExecutionSpace>>().num_neighbors_view());
    using IndexView2D = decltype(std::declval<Geometry<ExecutionSpace>>().surface_neighbors_view());

    // -------- Tags for different kernels --------
    struct planar                       {};
    struct planar_perturb               {};
    struct accumulate_surface_sources   {};
    struct solve_evaporation_term       {};
    struct solve_mixing_terms           {};
    struct solve_mixing                 {};
    struct solve_surface_mass_flux      {};
    struct surface_residual             {};
    struct perturbed_surface_residual   {};

    // -------- Stored views / data --------
    // Geometry-derived views
    SurfaceView surfaces;
    View2D      A_f;
    View2D      D_h;
    View1D      dz;
    IndexView1D num_neighbors;
    IndexView2D neighbor_list;

    // State views
    View2D P, X, W_l, W_v, h_l, evap, alpha, lhr, gk;
    View1D SS_l, SS_v, SS_m, CF_SS, TM_SS, VD_SS;
    View1D G_l_tm, G_v_tm, Q_m_tm, M_m_tm;
    View1D G_l_vd, G_v_vd, Q_m_vd, M_m_vd;

    // Fluid / constants
    double Tsat, h_f, h_fg, h_g, rho_f, rho_g, mu_f, mu_g, v_f, v_fg, v_g, sigma;

    Water  fluid;
    size_t k;       // surface_plane
    size_t k_node;  // node_plane
    double gap_width;
    double aspect;
    double K_ns{};
    double gtol{};
    double tol{};
    size_t max_inner_iter = 50;
    size_t max_outer_iter = 25;
    size_t current_ns1;
    double current_dG;

    // Views for mixing terms
    View1D gbar0;
    View1D reyn0;
    View1D Theta;

    // Views for perturbation loop residuals and Jacobian
    View1D f0;
    View1D f3;
    View2D dfdg;

    ANTSFunctor(StateType& state)
        : surfaces(state.geom->surface_view())
        , A_f(state.geom->channel_area_view())
        , D_h(state.geom->hydraulic_diameter_view())
        , dz(state.geom->dz_view())
        , num_neighbors(state.geom->num_neighbors_view())
        , neighbor_list(state.geom->surface_neighbors_view())
        , P(state.P)
        , X(state.X)
        , W_l(state.W_l)
        , W_v(state.W_v)
        , h_l(state.h_l)
        , evap(state.evap)
        , alpha(state.alpha)
        , lhr(state.lhr)
        , gk(state.gk)
        , SS_l(state.SS_l)
        , SS_v(state.SS_v)
        , SS_m(state.SS_m)
        , CF_SS(state.CF_SS)
        , TM_SS(state.TM_SS)
        , VD_SS(state.VD_SS)
        , G_l_tm(state.G_l_tm)
        , G_v_tm(state.G_v_tm)
        , Q_m_tm(state.Q_m_tm)
        , M_m_tm(state.M_m_tm)
        , G_l_vd(state.G_l_vd)
        , G_v_vd(state.G_v_vd)
        , Q_m_vd(state.Q_m_vd)
        , M_m_vd(state.M_m_vd)
        , Tsat(state.fluid.Tsat())
        , h_f(state.fluid.h_f())
        , h_fg(state.fluid.h_fg())
        , h_g(state.fluid.h_g())
        , rho_f(state.fluid.rho_f())
        , rho_g(state.fluid.rho_g())
        , mu_f(state.fluid.mu_f())
        , mu_g(state.fluid.mu_g())
        , v_f(state.fluid.v_f())
        , v_fg(state.fluid.v_fg())
        , v_g(state.fluid.v_g())
        , sigma(state.fluid.sigma())
        , fluid(state.fluid)
        , k(state.surface_plane)
        , k_node(state.node_plane)
        , gap_width(state.geom->gap_width())
        , aspect(state.geom->aspect_ratio())
    {
        const size_t nchan = A_f.extent(0);
        gbar0 = View1D("gbar0", nchan);
        reyn0 = View1D("reyn0", nchan);
        Theta = View1D("Theta", nchan);

        const size_t nsurf = surfaces.extent(0);
        f0   = View1D("f0", nsurf);
        f3   = View1D("f3", nsurf);
        dfdg = View2D("dfdg", nsurf, nsurf);

        // initialize source terms to 0.0
        Kokkos::deep_copy(SS_l, 0.0);
        Kokkos::deep_copy(SS_v, 0.0);
        Kokkos::deep_copy(SS_m, 0.0);
        Kokkos::deep_copy(CF_SS, 0.0);
        Kokkos::deep_copy(TM_SS, 0.0);
        Kokkos::deep_copy(VD_SS, 0.0);
    }

    // helper functions to execute operator kernels
    void accumulate_surf_sources() {
        // zero source terms
        Kokkos::deep_copy(SS_l,  0.0);
        Kokkos::deep_copy(SS_v,  0.0);
        Kokkos::deep_copy(SS_m,  0.0);
        Kokkos::deep_copy(CF_SS, 0.0);
        Kokkos::deep_copy(TM_SS, 0.0);
        Kokkos::deep_copy(VD_SS, 0.0);

        using policy_type = Kokkos::RangePolicy<ExecutionSpace, accumulate_surface_sources>;
        const size_t nsurf = surfaces.extent(0);

        Kokkos::parallel_for("TH::accumulate_surface_sources", policy_type(0, nsurf), *this);
    }

    // --------- One operator per high-level routine ---------

    // TH::planar -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void operator()(planar, const size_t ij) const {
        TH::solve_flow_rates<ExecutionSpace>(ij, k, k_node, A_f(ij, k), dz(k_node), evap, SS_l, SS_v, W_l, W_v);
        TH::solve_enthalpy<ExecutionSpace>(ij, k, k_node, dz(k_node), gap_width, h_g, W_l, W_v, lhr, SS_m, h_l);
        TH::solve_void_fraction<ExecutionSpace>(ij, k, k_node, A_f(ij, k), D_h(ij, k), rho_f, rho_g, h_f,
            h_fg, mu_g, sigma, max_inner_iter, fluid, P, W_l, W_v, h_l, X, alpha);
        TH::solve_quality<ExecutionSpace>(ij, k, k_node, A_f(ij, k), W_l, W_v, X);
        TH::solve_pressure<ExecutionSpace>(ij, k, k_node, A_f(ij, k), D_h(ij, k), dz(k_node), rho_f, rho_g, mu_f,
            mu_g, fluid, W_l, W_v, h_l, X, alpha, CF_SS, TM_SS, VD_SS, P);
    }

    // TH::planar_perturb -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void operator()(planar_perturb, const size_t ij) const {
        // Similar to planar, but with perturbation logic.
    }

    // TH::accumulate_surface_sources -> per-surface (ns)
    KOKKOS_INLINE_FUNCTION
    void operator()(accumulate_surface_sources, const size_t ns) const {
        auto surf = surfaces(ns);
        size_t i = surf.from_node;
        size_t j = surf.to_node;
        size_t i_donor = (gk(ns, k_node) >= 0) ? i : j;

        double A_f_i = A_f(i, k);
        double A_f_j = A_f(j, k);
        double A_f_donor = A_f(i_donor, k-1);

        double sl = gk(ns, k_node) * (1.0 - X(i_donor, k-1)) + G_l_tm(ns) + G_l_vd(ns);
        double sl_term = gap_width * sl;
        Kokkos::atomic_add(&SS_l(i), sl_term);
        Kokkos::atomic_add(&SS_l(j), -sl_term);

        double sv = gk(ns, k_node) * X(i_donor, k-1) + G_v_tm(ns) + G_v_vd(ns);
        double sv_term = gap_width * sv;
        Kokkos::atomic_add(&SS_v(i), sv_term);
        Kokkos::atomic_add(&SS_v(j), -sv_term);

        double h_l_donor = h_l(i_donor, k-1);
        double term = gap_width * (gk(ns, k_node) * h_l_donor + Q_m_tm(ns) + Q_m_vd(ns));
        Kokkos::atomic_add(&SS_m(i), term);
        Kokkos::atomic_add(&SS_m(j), -term);

        // Compute V_m for donor channel inline
        double v_m_donor;
        if (alpha(i_donor, k-1) < 1e-6) {
            v_m_donor = 1.0 / rho_f;
        } else if (alpha(i_donor, k-1) > 1.0 - 1e-6) {
            v_m_donor = 1.0 / rho_g;
        } else {
            double X_donor = X(i_donor, k-1);
            double rho_l_donor = fluid.rho(h_l_donor);
            v_m_donor = (1.0 - X_donor) * (1.0 - X_donor) / ((1.0 - alpha(i_donor, k-1)) * rho_l_donor) +
                        X_donor * X_donor / (alpha(i_donor, k-1) * rho_g);
        }
        double W_m_donor = W_l(i_donor, k-1) + W_v(i_donor, k-1);
        double V_m_donor = v_m_donor * W_m_donor / A_f_donor;

        double cf_term = gap_width * gk(ns, k_node) * V_m_donor;
        Kokkos::atomic_add(&CF_SS(i), cf_term);
        Kokkos::atomic_add(&CF_SS(j), -cf_term);

        double tm_term = gap_width * M_m_tm(ns);
        Kokkos::atomic_add(&TM_SS(i), tm_term);
        Kokkos::atomic_add(&TM_SS(j), -tm_term);

        double vd_term = gap_width * M_m_vd(ns);
        Kokkos::atomic_add(&VD_SS(i), vd_term);
        Kokkos::atomic_add(&VD_SS(j), -vd_term);
    }

    // TH::solve_evaporation_term -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void operator()(solve_evaporation_term, const size_t ij) const {
        const double H_0 = 0.075; // [s^-1 K^-1], condensation parameter

        double P_H = 4.0 * A_f(ij, k) / D_h(ij, k); // heated perimeter [m]
        double Re = (W_l(ij, k) / A_f(ij, k)) * D_h(ij, k) / fluid.mu(h_l(ij, k)); // Reynolds number
        double Pr = fluid.Cp(h_l(ij, k)) * fluid.mu(h_l(ij, k)) / fluid.k(h_l(ij, k)); // Prandtl number
        double Pe = Re * Pr; // Peclet number
        double Qflux_wall = lhr(ij, k_node) / P_H; // wall heat flux [W/m^2]
        double G_l = W_l(ij, k) / A_f(ij, k); // liquid mass flux [kg/m^2-s]
        double G_v = W_v(ij, k) / A_f(ij, k); // vapor mass flux [kg/m^2-s]
        double G_m = G_l + G_v; // mixture mass flux [kg/m^2-s]

        // Void departure (Eq. 52 from ANTS Theory)
        double void_dc = (Pe < 70000.0) ? (0.0022 * Pe * (Qflux_wall / G_m)) : (154.0 * (Qflux_wall / G_m));

        double Qflux_boil; // boiling heat flux [W/m]
        if (h_l(ij, k) < h_f) {
            // Subcooled region
            if ((h_f - h_l(ij, k)) < void_dc) {
                Qflux_boil = Qflux_wall * (1.0 - ((h_f - h_l(ij, k)) / void_dc));
            } else {
                Qflux_boil = 0.0;
            }

            double epsilon = fluid.rho(h_l(ij, k)) * (h_f - h_l(ij, k)) / (rho_g * h_fg); // pumping parameter (Eq. 53)
            double gamma_cond = (H_0 * (1.0 / v_fg) * A_f(ij, k) * alpha(ij, k) * (Tsat - fluid.T(h_l(ij, k)))) / P_H; // condensation rate (Eq. 54)
            evap(ij, k_node) = P_H * Qflux_boil / (h_fg * (1.0 + epsilon)) - P_H * gamma_cond; // Eq. 50

        } else {
            // Saturated region
            Qflux_boil = Qflux_wall;
            evap(ij, k_node) = P_H * Qflux_boil / h_fg;
        }
    }

    // TH::solve_mixing_terms -> per-channel to setup terms needed for 'solve_mixing' per-surface calculations
    KOKKOS_INLINE_FUNCTION
    void operator()(solve_mixing_terms, const size_t ij) const {
        const double Thetam = 5.0; // constant set equal to 5.0 for BWR applications, from ANTS Theory

        if (A_f(ij, k) < 1e-12) {
            gbar0(ij) = 0.0;
            reyn0(ij) = 0.0;
            Theta(ij) = 1.0;
            return;
        }

        double viscmi = X(ij, k) / mu_g + (1.0 - X(ij, k)) / mu_f;
        gbar0(ij) = (W_l(ij, k) + W_v(ij, k)) / A_f(ij, k);

        // Protect against zero flow
        if (gbar0(ij) < 1e-6) {
            reyn0(ij) = 0.0;
            Theta(ij) = 1.0;
            return;
        }

        reyn0(ij) = gbar0(ij) * D_h(ij, k) / viscmi;

        double Xmm = (0.4 * Kokkos::sqrt(rho_f * (rho_f - rho_g) * g * D_h(ij, k)) / gbar0(ij) + 0.6) / (Kokkos::sqrt(rho_f / rho_g) + 0.6);
        double X0m = 0.57 * Kokkos::pow(reyn0(ij), 0.0417);
        double Xfm = X(ij, k) / Xmm;
        if (X(ij, k) < Xmm) {
            Theta(ij) = 1.0 + (Thetam - 1.0) * Xfm;
        } else {
            Theta(ij) = 1.0 + (Thetam - 1.0) * (1.0 - X0m) / (Xfm - X0m);
        }

    }

    // TH::solve_mixing -> per-surface (ns)
    KOKKOS_INLINE_FUNCTION
    void operator()(solve_mixing, const size_t ns) const {

        Surface surf = surfaces(ns);
        size_t i = surf.from_node;
        size_t j = surf.to_node;

        double A_f_i = A_f(i, k);
        double A_f_j = A_f(j, k);
        double D_h_i = D_h(i, k);
        double D_h_j = D_h(j, k);

        // Compute heated perimeter and rod diameter inline
        double P_h_i = 4.0 * A_f_i / D_h_i;  // heated perimeter
        double P_h_j = 4.0 * A_f_j / D_h_j;
        double D_rod_i = P_h_i / M_PI;
        double D_rod_j = P_h_j / M_PI;

        // Skip mixing for channels with insufficient flow
        const double flow_threshold = 1e-6;
        if (gbar0(i) < flow_threshold || gbar0(j) < flow_threshold) {
            G_l_tm(ns) = 0.0;
            G_v_tm(ns) = 0.0;
            Q_m_tm(ns) = 0.0;
            M_m_tm(ns) = 0.0;
            G_l_vd(ns) = 0.0;
            G_v_vd(ns) = 0.0;
            Q_m_vd(ns) = 0.0;
            M_m_vd(ns) = 0.0;
            return;
        }

        // Compute mixture mass flux
        double G_m_i = (W_l(i, k) + W_v(i, k)) / A_f_i;
        double G_m_j = (W_l(j, k) + W_v(j, k)) / A_f_j;
        double rho_l_i = fluid.rho(h_l(i, k));
        double rho_l_j = fluid.rho(h_l(j, k));
        double h_l_i = h_l(i, k);
        double h_l_j = h_l(j, k);
        double alpha_i = alpha(i, k);
        double alpha_j = alpha(j, k);

        // Liquid velocity inline calculation
        double V_l_i = __liquid_velocity(W_l(i, k), A_f_i, alpha_i, rho_l_i);
        double V_l_j = __liquid_velocity(W_l(j, k), A_f_j, alpha_j, rho_l_j);


        // Vapor velocity inline calculation
        double V_v_i = __vapor_velocity(W_v(i, k), A_f_i, alpha_i, rho_g);
        double V_v_j = __vapor_velocity(W_v(j, k), A_f_j, alpha_j, rho_g);

        // Quality average with protection against division by zero
        const double K_M = 1.4;
        double G_m_sum = G_m_i + G_m_j;
        double X_bar = 0.0;
        if (G_m_sum > 1e-6) {
            X_bar = K_M * (G_m_i - G_m_j) / G_m_sum;
        }

        double tp_mult = 0.5 * (Theta(i) + Theta(j));
        double lambda = 0.0058 * gap_width / D_rod_i;
        double reynbar = 0.5 * (reyn0(i) + reyn0(j));

        double spbar = 0.5 * (fluid.mu(h_l(i, k)) + fluid.mu(h_l(j, k)));
        double eddy_V;
        if (reyn0(i) < reyn0(j)) {
            eddy_V = 0.5 * lambda * Kokkos::pow(reynbar, -0.1) * (1.0 + Kokkos::pow(D_h_i / D_h_j, 1.5)) * (D_h_i / D_rod_i) * gbar0(i) * spbar;
        } else {
            eddy_V = 0.5 * lambda * Kokkos::pow(reynbar, -0.1) * (1.0 + Kokkos::pow(D_h_j / D_h_i, 1.5)) * (D_h_j / D_rod_j) * gbar0(j) * spbar;
        }

        // Limit eddy diffusivity to prevent numerical instabilities
        // Typical values should be O(0.001 to 0.1 m³/s)
        if (eddy_V > 0.5) eddy_V = 0.5;
        if (eddy_V < 0.0) eddy_V = 0.0;

        // Turbulent mixing liquid mass transfer (Eq. 37)
        G_l_tm(ns) = eddy_V * tp_mult * ((1 - alpha_i) * rho_l_i - (1 - alpha_j) * rho_l_j);

        // Turbulent mixing vapor mass transfer (Eq. 38)
        G_v_tm(ns) = eddy_V * tp_mult * rho_g * (alpha_i - alpha_j);

        // Turbulent mixing energy transfer (Eq. 39)
        Q_m_tm(ns) = eddy_V * tp_mult * (
            (1 - alpha_i) * rho_l_i * h_l_i + alpha_i * rho_g * h_g
            - (1 - alpha_j) * rho_l_j * h_l_j - alpha_j * rho_g * h_g
        );

        // Turbulent mixing momentum transfer (Eq. 40)
        M_m_tm(ns) = eddy_V * tp_mult * (G_m_i - G_m_j);

        // Void drift liquid mass transfer (Eq. 42)
        G_l_vd(ns) = eddy_V * tp_mult * X_bar * (alpha_i * rho_l_i + alpha_j * rho_l_j);

        // Void drift vapor mass transfer (Eq. 41)
        G_v_vd(ns) = -eddy_V * tp_mult * X_bar * (alpha_i + alpha_j) * rho_g;

        // Void drift energy transfer (Eq. 43)
        Q_m_vd(ns) = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i * h_l_i + alpha_j * rho_l_j * h_l_j
            - (alpha_i + alpha_j) * rho_g * h_g
        );

        // Void drift momentum transfer (Eq. 44)
        M_m_vd(ns) = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i * V_l_i + alpha_j * rho_l_j * V_l_j
            - (alpha_i * V_v_i + alpha_j * V_v_j) * rho_g
        );

    }

    // TH::solve_surface_mass_flux -> per-surface (ns)
    KOKKOS_INLINE_FUNCTION
    void operator()(solve_surface_mass_flux, const size_t ns) const {
        // Use P, X, gk, surfaces, num_neighbors, neighbor_list, etc.
        // to compute/update crossflow gk(ns,k_node).
    }

    // TH::solve_surface_mass_flux - surface_residual -> per-surface (ns)
    KOKKOS_INLINE_FUNCTION
    void operator()(surface_residual, const size_t ns) const {
        const double K_ns = 0.5; // gap loss coefficient
        size_t i = surfaces(ns).from_node;
        size_t j = surfaces(ns).to_node;
        size_t i_donor = (gk(ns, k_node) >= 0) ? i : j;
        double rho_m = fluid.rho_m(X(i_donor, k));
        double deltaP = P(i, k) - P(j, k); // Eq. 56 from ANTS Theory
        double Fns = 0.5 * K_ns * gk(ns, k_node) * Kokkos::abs(gk(ns, k_node)) / rho_m; // Eq. 57 from ANTS Theory
        f0(ns) = -dz(k_node) * aspect * (deltaP - Fns); // Eq. 55 from ANTS Theory
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(perturbed_surface_residual, const size_t n) const {
        const double K_ns = 0.5; // gap loss coefficient
        const size_t ns1 = current_ns1;         // constant for this kernel launch
        const size_t ns  = neighbor_list(ns1, n);

        const size_t i = surfaces(ns).from_node;
        const size_t j = surfaces(ns).to_node;
        const size_t i_donor = (gk(ns, k_node) >= 0.0) ? i : j;

        const double rho_m  = fluid.rho_m(X(i_donor, k));
        const double deltaP = P(i, k) - P(j, k);
        const double Gj     = gk(ns, k_node);
        const double Fns    = 0.5 * K_ns * Gj * Kokkos::abs(Gj) / rho_m;

        f3(ns) = -dz(k_node) * aspect * (deltaP - Fns);
        dfdg(ns, ns1) = (f3(ns) - f0(ns)) / current_dG;
    }
};

} // namespace TH
