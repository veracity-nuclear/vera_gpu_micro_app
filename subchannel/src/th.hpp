#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <utility>
#include <type_traits>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

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
struct ANTSFunctor {
    using StateType = State<ExecutionSpace>;
    using View1D    = typename StateType::View1D; // double*
    using View2D    = typename StateType::View2D; // double**

    // Geometry view types deduced from Geometry<ExecutionSpace>
    using SurfaceView = decltype(std::declval<Geometry<ExecutionSpace>>().surface_view());
    using IndexView1D = decltype(std::declval<Geometry<ExecutionSpace>>().num_neighbors_view());
    using IndexView2D = decltype(std::declval<Geometry<ExecutionSpace>>().surface_neighbors_view());
    using JacPosView  = decltype(std::declval<Geometry<ExecutionSpace>>().jacobian_position_view());

    using team_member = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;

    // -------- Tags for different kernels --------
    struct planar                       {};
    struct accumulate_surf_sources      {};
    struct solve_evaporation_term       {};
    struct solve_mixing_terms           {};
    struct solve_mixing                 {};
    struct surface_residual             {};
    struct jacobian_columns             {};
    struct jacobian_columns_analytic    {};
    struct channel_partials             {};
    struct copy_gk_from_previous_plane  {};
    struct apply_gk_update              {};

    // -------- Stored views / data --------
    // Geometry-derived views
    SurfaceView surfaces;
    View2D      A_f;
    View2D      D_h;
    View1D      dz;
    IndexView1D num_neighbors;
    IndexView2D neighbor_list;
    JacPosView  jac_pos;        // (ns1, n) -> CSR position of Jacobian entry (row neighbor(ns1,n), col ns1)

    // State views
    View2D P, X, W_l, W_v, h_l, evap, alpha, lhr, gk;
    View1D SS_l, SS_v, SS_m, CF_SS, TM_SS, VD_SS;
    View1D G_l_tm, G_v_tm, Q_m_tm, M_m_tm;
    View1D G_l_vd, G_v_vd, Q_m_vd, M_m_vd;

    // Fluid / constants
    double Tsat, h_f, h_fg, h_g, rho_f, rho_g, mu_f, mu_g, v_f, v_fg, v_g, sigma;

    Water  fluid;
    size_t nchan;
    size_t nsurf;
    size_t k;       // surface_plane
    size_t k_node;  // node_plane
    double gap_width;
    double aspect;
    const double K_ns = 0.5; // gap loss coefficient
    const double gtol = 1e-3;
    const double tol = 1e-8;
    size_t max_inner_iter = 50;
    size_t max_outer_iter = 25;

    // Views for mixing terms
    View1D gbar0;
    View1D reyn0;
    View1D Theta;

    // Residual vector and sparse (CSR) Jacobian values; the CSR structure lives in Geometry
    View1D f0;
    View1D jac_values;
    View1D jac_values_fd;   // finite-difference Jacobian, only filled in "check" mode

    // Per-channel partial derivatives of the planar map with respect to the channel's accumulated
    // sources, filled by the channel_partials kernel and consumed by jacobian_columns_analytic:
    //   dchan(ij, 0) = dP/dSS_l, dchan(ij, 1) = dP/dSS_v, dchan(ij, 2) = dP/dCF_SS,
    //   dchan(ij, 3) = dX/dSS_l, dchan(ij, 4) = dX/dSS_v
    // (dP/dSS_m and dX/dSS_m vanish because the fluid properties here do not depend on enthalpy.)
    View2D dchan;
    int debug_channel = -1;      // if >= 0, channel_partials prints an FD check of dP/dWl, dP/dWv for this channel
    int jacobian_mode = 0;       // 0 analytic, 1 finite difference, 2 both
    bool void_bisection = false; // original bisection instead of safeguarded Newton for the void fraction
    bool fd_consistent_X = false; // FD columns recompute X from the perturbed flow rates (true planar map) instead of the stored X

    ANTSFunctor(StateType& state)
        : surfaces(state.geom->surface_view())
        , A_f(state.geom->channel_area_view())
        , D_h(state.geom->hydraulic_diameter_view())
        , dz(state.geom->dz_view())
        , num_neighbors(state.geom->num_neighbors_view())
        , neighbor_list(state.geom->surface_neighbors_view())
        , jac_pos(state.geom->jacobian_position_view())
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
        , nchan(state.geom->nchannels())
        , nsurf(state.geom->nsurfaces())
        , k(state.surface_plane)
        , k_node(state.node_plane)
        , gap_width(state.geom->gap_width())
        , aspect(state.geom->aspect_ratio())
    {
        gbar0 = View1D("gbar0", nchan);
        reyn0 = View1D("reyn0", nchan);
        Theta = View1D("Theta", nchan);

        f0   = View1D("f0", nsurf);
        jac_values = View1D("jac_values", state.geom->jacobian_nnz());
        jacobian_mode = state.jacobian_mode;
        void_bisection = state.void_bisection;
        if (jacobian_mode == 2) jac_values_fd = View1D("jac_values_fd", state.geom->jacobian_nnz());
        dchan = View2D("dchan", nchan, 5);
    }

    // helper functions to execute operator kernels
    void accumulate_surface_sources() {

        // zero source terms
        Kokkos::deep_copy(SS_l,  0.0);
        Kokkos::deep_copy(SS_v,  0.0);
        Kokkos::deep_copy(SS_m,  0.0);
        Kokkos::deep_copy(CF_SS, 0.0);
        Kokkos::deep_copy(TM_SS, 0.0);
        Kokkos::deep_copy(VD_SS, 0.0);

        using policy = Kokkos::RangePolicy<ExecutionSpace, accumulate_surf_sources>;
        Kokkos::parallel_for("TH::accumulate_surface_sources", policy(0, nsurf), *this);
    }

    // --------- One operator per high-level routine ---------

    // TH::planar -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void operator()(planar, const size_t ij) const {
        solve_flow_rates(ij, W_l(ij, k), W_v(ij, k));
        solve_enthalpy(ij, W_l(ij, k), W_v(ij, k), h_l(ij, k));
        solve_void_fraction(ij, W_l(ij, k), W_v(ij, k), h_l(ij, k), alpha(ij, k));
        solve_quality(ij, W_l(ij, k), W_v(ij, k), h_l(ij, k), X(ij, k));
        solve_pressure(ij, W_l(ij, k), W_v(ij, k), h_l(ij, k), alpha(ij, k), X(ij, k), P(ij, k));
    }

    // Source contributions of surface ns for a given crossflow value gk_ns: the six terms that are added
    // to the from-node and subtracted from the to-node (liquid mass, vapor mass, energy, crossflow
    // momentum, turbulent-mixing momentum, void-drift momentum). Shared by the source accumulation
    // and by the Jacobian kernel, which evaluates it at the perturbed gk.
    KOKKOS_INLINE_FUNCTION
    void surface_source_terms(const size_t ns, const double gk_ns, double terms[6]) const {
        auto surf = surfaces(ns);
        size_t i = surf.from_node;
        size_t j = surf.to_node;
        size_t i_donor = (gk_ns >= 0) ? i : j;

        double A_f_donor = A_f(i_donor, k-1);

        double sl = gk_ns * (1.0 - X(i_donor, k-1)) + G_l_tm(ns) + G_l_vd(ns);
        terms[0] = gap_width * sl;

        double sv = gk_ns * X(i_donor, k-1) + G_v_tm(ns) + G_v_vd(ns);
        terms[1] = gap_width * sv;

        double h_l_donor = h_l(i_donor, k-1);
        terms[2] = gap_width * (gk_ns * h_l_donor + Q_m_tm(ns) + Q_m_vd(ns));

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

        terms[3] = gap_width * gk_ns * V_m_donor;
        terms[4] = gap_width * M_m_tm(ns);
        terms[5] = gap_width * M_m_vd(ns);
    }

    // TH::accumulate_surface_sources -> per-surface (ns)
    KOKKOS_INLINE_FUNCTION
    void operator()(accumulate_surf_sources, const size_t ns) const {
        auto surf = surfaces(ns);
        size_t i = surf.from_node;
        size_t j = surf.to_node;

        double t[6];
        surface_source_terms(ns, gk(ns, k_node), t);

        Kokkos::atomic_add(&SS_l(i), t[0]);
        Kokkos::atomic_add(&SS_l(j), -t[0]);
        Kokkos::atomic_add(&SS_v(i), t[1]);
        Kokkos::atomic_add(&SS_v(j), -t[1]);
        Kokkos::atomic_add(&SS_m(i), t[2]);
        Kokkos::atomic_add(&SS_m(j), -t[2]);
        Kokkos::atomic_add(&CF_SS(i), t[3]);
        Kokkos::atomic_add(&CF_SS(j), -t[3]);
        Kokkos::atomic_add(&TM_SS(i), t[4]);
        Kokkos::atomic_add(&TM_SS(j), -t[4]);
        Kokkos::atomic_add(&VD_SS(i), t[5]);
        Kokkos::atomic_add(&VD_SS(j), -t[5]);
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

    // TH::solve_surface_mass_flux - surface_residual -> per-surface (ns)
    KOKKOS_INLINE_FUNCTION
    void operator()(surface_residual, const size_t ns) const {
        size_t i = surfaces(ns).from_node;
        size_t j = surfaces(ns).to_node;
        size_t i_donor = (gk(ns, k_node) >= 0) ? i : j;
        double rho_m = fluid.rho_m(X(i_donor, k));
        double deltaP = P(i, k) - P(j, k); // Eq. 56 from ANTS Theory
        double Fns = 0.5 * K_ns * gk(ns, k_node) * Kokkos::abs(gk(ns, k_node)) / rho_m; // Eq. 57 from ANTS Theory
        f0(ns) = -dz(k_node) * aspect * (deltaP - Fns); // Eq. 55 from ANTS Theory
    }

    // Planar solution of channel ij with the accumulated sources shifted by dSS (same six-term order as
    // surface_source_terms). Used by the Jacobian kernel; with dSS = 0 it reproduces TH::planar.
    KOKKOS_INLINE_FUNCTION
    void channel_state_with_source_shift(const size_t ij, const double dSS[6], double& X_ij, double& P_ij) const {
        double Wl, Wv, hl;
        double alpha_ij = alpha(ij, k); // keep the stored value when the void solve leaves it untouched
        solve_flow_rates(ij, SS_l(ij) + dSS[0], SS_v(ij) + dSS[1], Wl, Wv);
        solve_enthalpy(ij, Wl, Wv, SS_m(ij) + dSS[2], hl);
        solve_void_fraction(ij, Wl, Wv, hl, alpha_ij);
        if (fd_consistent_X) {
            X_ij = Wv / (Wv + Wl);
        } else {
            solve_quality(ij, Wl, Wv, hl, X_ij);
        }
        solve_pressure(ij, Wl, Wv, hl, alpha_ij, X_ij, CF_SS(ij) + dSS[3], TM_SS(ij) + dSS[4], VD_SS(ij) + dSS[5], P_ij);
    }

    // TH::solve_surface_mass_flux - jacobian_columns -> per-surface (ns1)
    //
    // Column ns1 of the Jacobian d f / d gk by finite differences. Perturbing gk(ns1) only changes the
    // source terms of the two channels adjacent to ns1, so instead of re-accumulating every surface's
    // sources (O(nsurf) work per column, serialized in the original implementation) the perturbed
    // sources of those two channels are formed locally and only the residuals of the neighbor surfaces
    // of ns1 are re-evaluated. All columns are independent, so the whole Jacobian is one parallel kernel.
    KOKKOS_INLINE_FUNCTION
    void operator()(jacobian_columns, const size_t ns1) const {
        const double gk0 = gk(ns1, k_node);
        const double dG = (gk0 >= 0.0) ? -gtol : gtol;
        const size_t i1 = surfaces(ns1).from_node;
        const size_t j1 = surfaces(ns1).to_node;

        // Change in the sources of channels i1 (+d) and j1 (-d) due to the perturbation
        double base[6], pert[6], d[6], neg_d[6];
        const double zero[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        surface_source_terms(ns1, gk0, base);
        surface_source_terms(ns1, gk0 + dG, pert);
        for (int t = 0; t < 6; ++t) {
            d[t] = pert[t] - base[t];
            neg_d[t] = -d[t];
        }

        const size_t nneigh = num_neighbors(ns1);
        for (size_t n = 0; n < nneigh; ++n) {
            const size_t ns = neighbor_list(ns1, n);
            const size_t i = surfaces(ns).from_node;
            const size_t j = surfaces(ns).to_node;

            const double* dSS_i = (i == i1) ? d : ((i == j1) ? neg_d : zero);
            const double* dSS_j = (j == i1) ? d : ((j == j1) ? neg_d : zero);

            double X_i, P_i, X_j, P_j;
            channel_state_with_source_shift(i, dSS_i, X_i, P_i);
            channel_state_with_source_shift(j, dSS_j, X_j, P_j);

            const double G_cf = (ns == ns1) ? gk0 + dG : gk(ns, k_node);
            const double X_donor = (G_cf >= 0.0) ? X_i : X_j;
            const double rho_m = fluid.rho_m(X_donor);
            const double deltaP = P_i - P_j;
            const double Fns = 0.5 * K_ns * G_cf * Kokkos::abs(G_cf) / rho_m;
            const double f3 = -dz(k_node) * aspect * (deltaP - Fns);

            jac_values(jac_pos(ns1, n)) = (f3 - f0(ns)) / dG;
        }
    }

    // Starting guess for this plane's crossflow: the converged values of the plane below
    KOKKOS_INLINE_FUNCTION
    void operator()(copy_gk_from_previous_plane, const size_t ns) const {
        gk(ns, k_node) = gk(ns, k_node - 1);
    }

    // Newton update; f0 holds the linear-system solution at this point
    KOKKOS_INLINE_FUNCTION
    void operator()(apply_gk_update, const size_t ns) const {
        gk(ns, k_node) -= f0(ns);
    }

    // TH::solve_flow_rates -> per-channel (ij), with explicit liquid/vapor source terms
    KOKKOS_INLINE_FUNCTION
    void solve_flow_rates(size_t ij, double SSl, double SSv, double& Wl, double& Wv) const {

        // Update liquid flow rate (Eq. 61 from ANTS Theory)
        Wl = W_l(ij, k-1) - dz(k_node) * (evap(ij, k_node) + SSl);
        Wl = (Wl > 0.0) ? Wl : 1e-8; // prevent negative (making this 0.0 makes enthalpy nan)

        // Update vapor flow rate (Eq. 62 from ANTS Theory)
        Wv = W_v(ij, k-1) + dz(k_node) * (evap(ij, k_node) - SSv);
        Wv = (Wv > 0.0) ? Wv : 0.0; // prevent negative
    }

    // TH::solve_flow_rates -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void solve_flow_rates(size_t ij, double& Wl, double& Wv) const {
        solve_flow_rates(ij, SS_l(ij), SS_v(ij), Wl, Wv);
    }

    // TH::solve_enthalpy -> per-channel (ij), with explicit energy source term
    KOKKOS_INLINE_FUNCTION
    void solve_enthalpy(size_t ij, double Wl, double Wv, double SSm, double& hl) const {

        // Eq. 63 from ANTS Theory
        hl = (
            (W_v(ij, k-1) - Wv) * h_g
            + W_l(ij, k-1) * h_l(ij, k-1) + dz(k_node) * lhr(ij, k_node)
            - dz(k_node) * SSm
        ) / Wl;
    }

    // TH::solve_enthalpy -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void solve_enthalpy(size_t ij, double Wl, double Wv, double& hl) const {
        solve_enthalpy(ij, Wl, Wv, SS_m(ij), hl);
    }

    // TH::solve_void_fraction -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void solve_void_fraction(size_t ij, double Wl, double Wv, double hl, double& alpha) const {
        solve_void_fraction_impl(ij, Wl, Wv, hl, alpha, nullptr, nullptr);
    }

    // Drift-flux (Chexal-Lellouche form) void fraction: root of
    //   f(alpha) = (alpha C0 - 1) Gv + alpha C0 (rho_g/rho_l) Gl + alpha rho_g Vgj(alpha)
    // Solved with a bracketed Newton iteration (bisection step whenever Newton leaves the bracket or f' is
    // tiny); f'(alpha) is closed-form, so this converges in a handful of iterations instead of the ~50 of
    // the original pure bisection, to the same |f| < tol / bracket < tol criteria. With void_bisection set
    // the original bisection is used instead.
    // Optionally returns d alpha / d Wl and d alpha / d Wv by implicit differentiation at the root
    // (d alpha / d G = -f_G / f_alpha), including the dependence of B1 on the Reynolds number.
    KOKKOS_INLINE_FUNCTION
    void solve_void_fraction_impl(size_t ij, double Wl, double Wv, double hl, double& alpha,
                                  double* dalpha_dWl, double* dalpha_dWv) const {
        const double tol_local = 1e-8;
        const double eps = 1e-12; // small number to prevent division by zero
        if (dalpha_dWl) *dalpha_dWl = 0.0;
        if (dalpha_dWv) *dalpha_dWv = 0.0;

        // based on the Chexal-Lellouche drift flux model
        double pressure = P(0, 0); // assuming constant pressure for simplicity

        const double Af = A_f(ij, k);
        double Gv = Wv / Af; // vapor mass flux
        double Gl = Wl / Af; // liquid mass flux

        if (Gv < eps) {
            alpha = 0.0;
        }

        double rho_l = fluid.rho(hl);
        double mu_l = fluid.mu(hl);

        double Re_g = __Reynolds(Wv / Af, D_h(ij, k), mu_g); // local vapor Reynolds number
        double Re_f = __Reynolds(Wl / Af, D_h(ij, k), mu_l); // local liquid Reynolds number
        double Re = (Re_g > Re_f) ? Re_g : Re_f;
        double A1 = 1 / (1 + Kokkos::exp(-Re / 60000));
        double B1 = (0.8 < A1) ? 0.8 : A1; // from Zuber correlation
        double B2 = 1.41;

        // alpha-independent pieces of C0 and Vgj
        const double C1 = 4.0 * P_crit * P_crit / (pressure * (P_crit - pressure)); // Eq. 24 from ANTS Theory
        const double Lden = 1.0 - Kokkos::exp(-C1);
        const double K0 = B1 + (1 - B1) * Kokkos::pow(rho_g / rho_f, 0.25); // Eq. 25 from ANTS Theory
        const double r = (1 + 1.57 * (rho_g / rho_f)) / (1 - B1); // Eq. 26 from ANTS Theory
        const double Vgj0 = B2 * Kokkos::pow(((rho_f - rho_g) * g * sigma) / (rho_f * rho_f), 0.25); // Eq. 28 from ANTS Theory
        const double Gsum = Gv + (rho_g / rho_l) * Gl;

        // f and f' at alpha_val (f' only when requested)
        auto evaluate = [&](double alpha_val, double* fprime) {
            const double e = Kokkos::exp(-C1 * alpha_val);
            const double L = (1.0 - e) / Lden; // Eq. 23 from ANTS Theory
            const double ar = Kokkos::pow(alpha_val, r);
            const double D = K0 + (1 - K0) * ar;
            const double C0 = L / D; // Eq. 22 from ANTS Theory
            const double om = 1.0 - alpha_val;
            const double Vgj = Vgj0 * Kokkos::pow(om, B1); // Eq. 27 from ANTS Theory
            if (fprime) {
                const double Lp = C1 * e / Lden;
                const double Dp = (alpha_val > 0.0) ? (1 - K0) * r * ar / alpha_val : 0.0;
                const double C0p = (Lp * D - L * Dp) / (D * D);
                const double Vgjp = (om > 0.0) ? -B1 * Vgj / om : 0.0;
                *fprime = (C0 + alpha_val * C0p) * Gsum + rho_g * (Vgj + alpha_val * Vgjp);
            }
            return (alpha_val * C0 - 1.0) * Gv + alpha_val * C0 * (rho_g / rho_l) * Gl + alpha_val * rho_g * Vgj;
        };
        auto implicit_derivatives = [&](double alpha_val) {
            if (!dalpha_dWl && !dalpha_dWv) return;
            double fa_alpha;
            const double e = Kokkos::exp(-C1 * alpha_val);
            const double L = (1.0 - e) / Lden;
            const double ar = Kokkos::pow(alpha_val, r);
            const double D = K0 + (1 - K0) * ar;
            const double C0 = L / D;
            evaluate(alpha_val, &fa_alpha);
            if (Kokkos::fabs(fa_alpha) < 1e-300) return;
            // explicit dependence on the mass fluxes
            double f_Gv = alpha_val * C0 - 1.0;
            double f_Gl = alpha_val * C0 * (rho_g / rho_l);
            // dependence through B1(Re) when the Zuber limiter is not active (A1 < 0.8): K0, r and the
            // Vgj exponent all depend on B1
            if (A1 < 0.8) {
                const double dB1_dRe = A1 * (1.0 - A1) / 60000.0;
                const double dK0_dB1 = 1.0 - Kokkos::pow(rho_g / rho_f, 0.25);
                const double dr_dB1 = r / (1 - B1);
                const double lna = (alpha_val > 0.0) ? Kokkos::log(alpha_val) : 0.0;
                const double dD_dB1 = dK0_dB1 * (1.0 - ar) + (1 - K0) * ar * lna * dr_dB1;
                const double dC0_dB1 = -L * dD_dB1 / (D * D);
                const double Vgj = Vgj0 * Kokkos::pow(1.0 - alpha_val, B1);
                const double dVgj_dB1 = (alpha_val < 1.0) ? Vgj * Kokkos::log(1.0 - alpha_val) : 0.0;
                const double f_B1 = alpha_val * dC0_dB1 * Gsum + alpha_val * rho_g * dVgj_dB1;
                // Re = max(Re_g, Re_f)
                if (Re_g > Re_f) f_Gv += f_B1 * dB1_dRe * D_h(ij, k) / mu_g;
                else             f_Gl += f_B1 * dB1_dRe * D_h(ij, k) / mu_l;
            }
            if (dalpha_dWv) *dalpha_dWv = -(f_Gv / fa_alpha) / Af;
            if (dalpha_dWl) *dalpha_dWl = -(f_Gl / fa_alpha) / Af;
        };

        double a = 0.0;
        double b = 1.0;
        double fa = evaluate(a, nullptr);
        double fb = evaluate(b, nullptr);

        if (Kokkos::fabs(fa) < tol_local) {
            alpha = a;
            return;
        }
        if (Kokkos::fabs(fb) < tol_local) {
            alpha = b;
            return;
        }

        // Root must be bracketed
        if (fa * fb > 0) {
            // If not bracketed, use safer default (leave alpha unchanged)
            return;
        }

        if (void_bisection) {
            for (int i = 0; i < (int)max_inner_iter; i++) {
                double c = 0.5 * (a + b);
                double fc = evaluate(c, nullptr);
                if (Kokkos::fabs(fc) < tol_local || (b - a) < tol_local) {
                    alpha = c;
                    implicit_derivatives(c);
                    return;
                }
                if (fa * fc < 0) { b = c; fb = fc; } else { a = c; fa = fc; }
            }
            alpha = 0.5 * (a + b);
            implicit_derivatives(alpha);
            return;
        }

        // Safeguarded Newton. Start from the homogeneous estimate (C0 = 1, Vgj = 0), clipped into the bracket.
        double c = (Gsum > 0.0) ? Gv / Gsum : 0.5;
        if (!(c > a && c < b)) c = 0.5 * (a + b);
        for (int i = 0; i < (int)max_inner_iter; i++) {
            double fpc;
            const double fc = evaluate(c, &fpc);
            if (Kokkos::fabs(fc) < tol_local || (b - a) < tol_local) {
                alpha = c;
                implicit_derivatives(c);
                return;
            }
            // shrink the bracket around the root
            if (fa * fc < 0) { b = c; fb = fc; } else { a = c; fa = fc; }
            // Newton step, falling back to bisection if it leaves the bracket or the slope is unusable
            double cn = (Kokkos::fabs(fpc) > 1e-300) ? c - fc / fpc : 0.5 * (a + b);
            if (!(cn > a && cn < b)) cn = 0.5 * (a + b);
            c = cn;
        }

        alpha = 0.5 * (a + b);
        implicit_derivatives(alpha);
    }

    // TH::solve_quality -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void solve_quality(size_t ij, double Wl, double Wv, double hl, double& X) const {
        double G_v = W_v(ij, k) / A_f(ij, k); // vapor mass flux (Eq. 8 from ANTS Theory)
        double G_l = W_l(ij, k) / A_f(ij, k); // liquid mass flux (Eq. 9 from ANTS Theory)
        X = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory
    }

    // Partial derivatives of the plane-k pressure (as computed by solve_pressure) with respect to the
    // channel flow rates Wl and Wv, given the derivatives of alpha and X with respect to them. Mirrors
    // solve_pressure term by term; the gravity term and all property evaluations have zero derivative
    // because the fluid properties here are independent of enthalpy. Form losses are zero (K_loss = 0).
    KOKKOS_INLINE_FUNCTION
    void pressure_partials(size_t ij, double Wl, double Wv, double alpha_ijk, double X_ijk,
                           double dalpha_dWl, double dalpha_dWv, double dX_dWl, double dX_dWv,
                           double& dP_dWl, double& dP_dWv) const {
        const double a_1 = 0.1892;
        const double n = -0.2;
        const double Af = A_f(ij, k);
        const double Dh = D_h(ij, k);
        const double dzk = dz(k_node);

        const double G_l = Wl / Af;
        const double G = (Wl + Wv) / Af;
        const double dG = 1.0 / Af; // dG/dWl = dG/dWv

        // ----- acceleration: G^2 (nu_m(k) - nu_m(k-1)) -----
        double nu_m_k, dnu_dalpha = 0.0, dnu_dX = 0.0;
        if (alpha_ijk < 1e-6) {
            nu_m_k = 1.0 / rho_f;
        } else if (alpha_ijk > 1.0 - 1e-6) {
            nu_m_k = 1.0 / rho_g;
        } else {
            const double om = 1.0 - alpha_ijk, omX = 1.0 - X_ijk;
            const double rho_l = fluid.rho(0.0);
            nu_m_k = omX * omX / (om * rho_l) + X_ijk * X_ijk / (alpha_ijk * rho_g);
            dnu_dalpha = omX * omX / (om * om * rho_l) - X_ijk * X_ijk / (alpha_ijk * alpha_ijk * rho_g);
            dnu_dX = -2.0 * omX / (om * rho_l) + 2.0 * X_ijk / (alpha_ijk * rho_g);
        }
        double nu_m_km1;
        if (alpha(ij, k-1) < 1e-6) {
            nu_m_km1 = 1.0 / rho_f;
        } else if (alpha(ij, k-1) > 1.0 - 1e-6) {
            nu_m_km1 = 1.0 / rho_g;
        } else {
            nu_m_km1 = (1.0 - X(ij, k-1)) * (1.0 - X(ij, k-1)) / ((1.0 - alpha(ij, k-1)) * fluid.rho(h_l(ij, k-1))) +
                        X(ij, k-1) * X(ij, k-1) / (alpha(ij, k-1) * rho_g);
        }
        const double dnu_dWl = dnu_dalpha * dalpha_dWl + dnu_dX * dX_dWl;
        const double dnu_dWv = dnu_dalpha * dalpha_dWv + dnu_dX * dX_dWv;
        const double daccel_dWl = 2.0 * G * dG * (nu_m_k - nu_m_km1) + G * G * dnu_dWl;
        const double daccel_dWv = 2.0 * G * dG * (nu_m_k - nu_m_km1) + G * G * dnu_dWv;

        // ----- wall friction: K G^2 / (2 rho_f) phi2_ch -----
        const double mu_l = fluid.mu(0.0);
        const double Re = G_l * Dh / mu_l;
        const double f = a_1 * Kokkos::pow(Re, n);
        const double K = f * dzk / Dh;
        const double dK_dWl = (Re > 0.0) ? (n * f / Re) * (Dh / mu_l) * dG * dzk / Dh : 0.0;
        const double gamma = Kokkos::pow(rho_f / rho_g, 0.5) * Kokkos::pow(mu_g / mu_f, 0.2);
        double b;
        if (gamma <= 9.5) {
            b = 55.0 / Kokkos::sqrt(G);
        } else if (gamma < 28) {
            b = 520.0 / (gamma * Kokkos::sqrt(G));
        } else {
            b = 15000.0 / (gamma * gamma * Kokkos::sqrt(G));
        }
        const double db_dG = (G > 0.0) ? -0.5 * b / G : 0.0; // all branches scale as G^-1/2
        const double g2m1 = gamma * gamma - 1.0;
        double phi2, dphi2_db, dphi2_dX;
        if (X_ijk > 0.0 && X_ijk < 1.0) {
            const double x9 = Kokkos::pow(X_ijk, 0.9), y9 = Kokkos::pow(1.0 - X_ijk, 0.9);
            phi2 = 1.0 + g2m1 * (b * x9 * y9 + Kokkos::pow(X_ijk, 1.8));
            dphi2_db = g2m1 * x9 * y9;
            dphi2_dX = g2m1 * (b * 0.9 * (x9 * y9 / X_ijk - x9 * y9 / (1.0 - X_ijk)) + 1.8 * Kokkos::pow(X_ijk, 0.8));
        } else {
            // single phase (or all vapor): X is pinned, the X^0.9 slope is singular there, treat as constant
            phi2 = 1.0 + g2m1 * ((X_ijk >= 1.0) ? 1.0 : 0.0);
            dphi2_db = 0.0;
            dphi2_dX = 0.0;
        }
        const double dwall_dWl = (dK_dWl * G * G * phi2 + K * 2.0 * G * dG * phi2 + K * G * G * (dphi2_db * db_dG * dG + dphi2_dX * dX_dWl)) / (2.0 * rho_f);
        const double dwall_dWv = (K * 2.0 * G * dG * phi2 + K * G * G * (dphi2_db * db_dG * dG + dphi2_dX * dX_dWv)) / (2.0 * rho_f);

        // gravity and momentum-exchange terms do not depend on Wl, Wv
        dP_dWl = -(daccel_dWl + dwall_dWl);
        dP_dWv = -(daccel_dWv + dwall_dWv);
    }

    // TH::channel_partials -> per-channel (ij): derivatives of the planar map (P, X at plane k) with
    // respect to the channel's accumulated sources, evaluated at the current sources. One pass per
    // channel replaces the 14 re-evaluations per Jacobian column of the finite-difference scheme.
    KOKKOS_INLINE_FUNCTION
    void operator()(channel_partials, const size_t ij) const {
        const double dzk = dz(k_node);
        const double Af = A_f(ij, k);

        // flow rates and the clamps (a clamped rate has zero derivative)
        double Wl_raw = W_l(ij, k-1) - dzk * (evap(ij, k_node) + SS_l(ij));
        double Wv_raw = W_v(ij, k-1) + dzk * (evap(ij, k_node) - SS_v(ij));
        const double Wl = (Wl_raw > 0.0) ? Wl_raw : 1e-8;
        const double Wv = (Wv_raw > 0.0) ? Wv_raw : 0.0;
        const double dWl_dsl = (Wl_raw > 0.0) ? -dzk : 0.0;
        const double dWv_dsv = (Wv_raw > 0.0) ? -dzk : 0.0;

        double hl;
        solve_enthalpy(ij, Wl, Wv, SS_m(ij), hl);

        double alpha_ij = alpha(ij, k), dalpha_dWl, dalpha_dWv;
        solve_void_fraction_impl(ij, Wl, Wv, hl, alpha_ij, &dalpha_dWl, &dalpha_dWv);

        // quality X = Gv / (Gv + Gl) evaluated with the plane-k flow rates
        const double Gv = Wv / Af, Gl = Wl / Af, Gs = Gv + Gl;
        const double X_ij = Gv / Gs;
        const double dX_dWv = (Wv > 0.0) ? Gl / (Af * Gs * Gs) : 0.0;
        const double dX_dWl = (Wv > 0.0) ? -Gv / (Af * Gs * Gs) : 0.0;

        double dP_dWl, dP_dWv;
        pressure_partials(ij, Wl, Wv, alpha_ij, X_ij, dalpha_dWl, dalpha_dWv, dX_dWl, dX_dWv, dP_dWl, dP_dWv);

        if ((int)ij == debug_channel) {
            // finite-difference check of the pressure partials through the full planar chain (alpha, X, P)
            auto P_of = [&](double wl, double wv) {
                double a_ = alpha(ij, k), x_ = (wv + wl > 0.0) ? wv / (wv + wl) : 0.0, p_;
                solve_void_fraction_impl(ij, wl, wv, hl, a_, nullptr, nullptr);
                solve_pressure(ij, wl, wv, hl, a_, x_, CF_SS(ij), TM_SS(ij), VD_SS(ij), p_);
                return p_;
            };
            const double hwl = 1e-6 * (Kokkos::abs(Wl) + 1e-3), hwv = 1e-6 * (Kokkos::abs(Wv) + 1e-3);
            const double fd_l = (P_of(Wl + hwl, Wv) - P_of(Wl - hwl, Wv)) / (2 * hwl);
            const double fd_v = (P_of(Wl, Wv + hwv) - P_of(Wl, Wv - hwv)) / (2 * hwv);
            printf("  [channel %d] Wl=%.6e Wv=%.6e alpha=%.4e X=%.4e | dP/dWl an=%.6e fd=%.6e | dP/dWv an=%.6e fd=%.6e | dalpha/dWv=%.3e dX/dWv=%.3e\n",
                   (int)ij, Wl, Wv, alpha_ij, X_ij, dP_dWl, fd_l, dP_dWv, fd_v, dalpha_dWv, dX_dWv);
        }

        dchan(ij, 0) = dP_dWl * dWl_dsl;   // dP/dSS_l
        dchan(ij, 1) = dP_dWv * dWv_dsv;   // dP/dSS_v
        dchan(ij, 2) = -dzk;               // dP/dCF_SS
        dchan(ij, 3) = dX_dWl * dWl_dsl;   // dX/dSS_l
        dchan(ij, 4) = dX_dWv * dWv_dsv;   // dX/dSS_v
    }

    // Derivatives of the surface-ns source terms (surface_source_terms order) with respect to gk(ns):
    // sl = gap (gk (1 - X_d) + ...), sv = gap (gk X_d + ...), sm = gap (gk h_d + ...), cf = gap gk V_m,d
    // where the donor cell d is fixed by the sign of gk.
    KOKKOS_INLINE_FUNCTION
    void surface_source_derivatives(const size_t ns, const double gk_ns, double& dsl, double& dsv, double& dcf) const {
        auto surf = surfaces(ns);
        const size_t i_donor = (gk_ns >= 0) ? surf.from_node : surf.to_node;
        const double Xd = X(i_donor, k-1);
        dsl = gap_width * (1.0 - Xd);
        dsv = gap_width * Xd;
        double v_m_donor;
        if (alpha(i_donor, k-1) < 1e-6) {
            v_m_donor = 1.0 / rho_f;
        } else if (alpha(i_donor, k-1) > 1.0 - 1e-6) {
            v_m_donor = 1.0 / rho_g;
        } else {
            v_m_donor = (1.0 - Xd) * (1.0 - Xd) / ((1.0 - alpha(i_donor, k-1)) * fluid.rho(h_l(i_donor, k-1))) +
                        Xd * Xd / (alpha(i_donor, k-1) * rho_g);
        }
        const double V_m_donor = v_m_donor * (W_l(i_donor, k-1) + W_v(i_donor, k-1)) / A_f(i_donor, k-1);
        dcf = gap_width * V_m_donor;
    }

    // TH::solve_surface_mass_flux - jacobian_columns_analytic -> per-surface (ns1)
    //
    // Exact derivative of the planar residual map (chain rule through surface sources -> channel flow
    // rates -> void fraction (implicit) and quality -> pressure), using the per-channel partials from
    // channel_partials. Unlike the finite-difference column, this includes the response of the quality
    // in the Fns term, so the Newton path differs slightly while the converged solution does not.
    KOKKOS_INLINE_FUNCTION
    void operator()(jacobian_columns_analytic, const size_t ns1) const {
        const double gk0 = gk(ns1, k_node);
        const size_t i1 = surfaces(ns1).from_node;
        const size_t j1 = surfaces(ns1).to_node;
        double dsl, dsv, dcf;
        surface_source_derivatives(ns1, gk0, dsl, dsv, dcf);

        const size_t nneigh = num_neighbors(ns1);
        for (size_t n = 0; n < nneigh; ++n) {
            const size_t ns = neighbor_list(ns1, n);
            const size_t i = surfaces(ns).from_node;
            const size_t j = surfaces(ns).to_node;
            const double si = (i == i1) ? 1.0 : ((i == j1) ? -1.0 : 0.0);
            const double sj = (j == i1) ? 1.0 : ((j == j1) ? -1.0 : 0.0);

            const double dP_i = si * (dchan(i, 0) * dsl + dchan(i, 1) * dsv + dchan(i, 2) * dcf);
            const double dP_j = sj * (dchan(j, 0) * dsl + dchan(j, 1) * dsv + dchan(j, 2) * dcf);
            const double dX_i = si * (dchan(i, 3) * dsl + dchan(i, 4) * dsv);
            const double dX_j = sj * (dchan(j, 3) * dsl + dchan(j, 4) * dsv);

            const double G_cf = gk(ns, k_node);
            const bool donor_i = (G_cf >= 0.0);
            const double X_d = donor_i ? X(i, k) : X(j, k);
            const double dX_d = donor_i ? dX_i : dX_j;
            const double rho_m = fluid.rho_m(X_d);
            // Fns = 0.5 K gk |gk| / rho_m = 0.5 K gk |gk| (X v_g + (1 - X) v_f)
            double dFns = 0.5 * K_ns * G_cf * Kokkos::abs(G_cf) * (v_g - v_f) * dX_d;
            // d Fns / d gk = K |gk| / rho_m vanishes at zero crossflow, and without it the Jacobian is the
            // incidence form B^T D B whose null space is the cycle space of the surface graph (singular LU
            // at the first Newton step of every plane). The finite-difference column is implicitly
            // regularized by its secant slope 0.5 K (|gk + dG| |gk + dG| - gk |gk|) / dG ~ K (|gk| + dG/2);
            // use that same secant slope here so the exact Jacobian is invertible for the same reason.
            if (ns == ns1) dFns += K_ns * (Kokkos::abs(G_cf) + 0.5 * gtol) / rho_m;

            jac_values(jac_pos(ns1, n)) = -dz(k_node) * aspect * (dP_i - dP_j - dFns);
        }
    }

    // TH::solve_pressure -> per-channel (ij)
    KOKKOS_INLINE_FUNCTION
    void solve_pressure(size_t ij, double Wl, double Wv, double hl, double alpha_ijk, double X_ijk, double& P_ijk) const {
        solve_pressure(ij, Wl, Wv, hl, alpha_ijk, X_ijk, CF_SS(ij), TM_SS(ij), VD_SS(ij), P_ijk);
    }

    // TH::solve_pressure -> per-channel (ij), with explicit momentum source terms
    KOKKOS_INLINE_FUNCTION
    void solve_pressure(size_t ij, double Wl, double Wv, double hl, double alpha_ijk, double X_ijk,
                        double CF, double TM, double VD, double& P_ijk) const {
        // coefficients for Adams correlation from ANTS Theory
        const double a_1 = 0.1892;
        const double n = -0.2;

        // mass flux (liq. only)
        double G_l = Wl / A_f(ij, k);

        // mass flux (mixture)
        double G = (Wl + Wv) / A_f(ij, k);

        // ----- two-phase acceleration pressure drop -----
        double nu_m_k, nu_m_km1;
        if (alpha_ijk < 1e-6) {
            nu_m_k = 1.0 / rho_f;
        } else if (alpha_ijk > 1.0 - 1e-6) {
            nu_m_k = 1.0 / rho_g;
        } else {
            nu_m_k = (1.0 - X_ijk) * (1.0 - X_ijk) / ((1.0 - alpha_ijk) * fluid.rho(hl)) + X_ijk * X_ijk / (alpha_ijk * rho_g);
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
        double Re = G_l * D_h(ij, k) / fluid.mu(hl);

        // frictional pressure drop from wall shear
        double f = a_1 * Kokkos::pow(Re, n);
        double K = f * dz(k_node) / D_h(ij, k);
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
        double phi2_ch = 1.0 + (gamma * gamma - 1.0) * (b * Kokkos::pow(X_ijk, 0.9) * Kokkos::pow((1.0 - X_ijk), 0.9) + Kokkos::pow(X_ijk, 1.8));

        // two-phase wall shear pressure drop
        double dP_wall_shear = K * G * G / (2.0 * rho_f) * phi2_ch;

        // form loss coefficient (no form losses in this simple model)
        double K_loss = 0.0;

        // two-phase multiplier for form losses (homogeneous), Eq. 35 from ANTS Theory
        double phi2_hom = 1.0 + X_ijk * (rho_f / rho_g - 1.0);

        // two-phase geometry form loss pressure drop, Eq. 36 from ANTS Theory
        double dP_form = K_loss * G * G / (2.0 * rho_f) * phi2_hom;

        // two-phase frictional pressure drop, Eq. 36 from ANTS Theory
        double dP_tpfric = dP_wall_shear + dP_form;

        // ----- two-phase gravitational pressure drop -----
        double dP_grav = fluid.rho(hl) * g * dz(k_node);

        // ----- momentum exchange -----
        double dP_CF = dz(k_node) * CF;
        double dP_TM = dz(k_node) * TM;
        double dP_VD = dz(k_node) * VD;
        double dP_momexch = dP_CF + dP_TM + dP_VD;

        // ----- total pressure drop -----
        double dP_total = dP_accel + dP_tpfric + dP_grav + dP_momexch;
        P_ijk = P(ij, k-1) - dP_total;
    }

};

} // namespace TH
