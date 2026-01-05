#include "th.hpp"

/**
 * ANTS Theory refers to
 *
 * Kropaczek, D. J., Salko, R. K. Jr, Hizoum, B., & Collins, B. S. (2023, July).
 * Advanced two-phase subchannel method via non-linear iteration.
 * Nuclear Engineering and Design, 408, 112328.
 *
 * https://doi.org/10.1016/j.nucengdes.2023.112328
 */

template <typename ExecutionSpace>
void TH::planar(State<ExecutionSpace>& state) {
    solve_flow_rates<ExecutionSpace>(state);
    solve_enthalpy<ExecutionSpace>(state);
    solve_void_fraction<ExecutionSpace>(state);
    solve_quality<ExecutionSpace>(state);
    solve_pressure<ExecutionSpace>(state);
}

template <typename ExecutionSpace>
void TH::solve_evaporation_term(State<ExecutionSpace>& state) {

    typename State<ExecutionSpace>::View2D mu = state.fluid->mu(state.h_l); // dynamic viscosity [Pa-s]
    typename State<ExecutionSpace>::View2D rho = state.fluid->rho(state.h_l); // liquid density [kg/m^3]
    typename State<ExecutionSpace>::View2D cond = state.fluid->k(state.h_l); // thermal conductivity [W/m-K]
    typename State<ExecutionSpace>::View2D Cp = state.fluid->Cp(state.h_l); // specific heat [J/kg-K]
    typename State<ExecutionSpace>::View2D T = state.fluid->T(state.h_l); // temperature [K]

    // Device-based implementation
    auto W_l = state.W_l;
    auto W_v = state.W_v;
    auto h_l = state.h_l;
    auto alpha = state.alpha;
    auto lhr = state.lhr;
    auto evap = state.evap;
    auto hydraulic_diameter_view = state.geom->hydraulic_diameter_view();
    auto flow_area_view = state.geom->channel_area_view();

    const size_t k = state.surface_plane;
    const size_t k_node = state.node_plane;
    const size_t nchannels = state.geom->nchannels();

    // Fluid properties (constants)
    const double h_f = state.fluid->h_f();
    const double h_fg = state.fluid->h_fg();
    const double rho_g = state.fluid->rho_g();
    const double v_fg = state.fluid->v_fg();
    const double Tsat = state.fluid->Tsat();
    const double H_0 = 0.075; // [s^-1 K^-1], condensation parameter

    // Geometry accessors (need to capture these outside lambda)
    auto geom = state.geom;

    // Compute evaporation term on device
    Kokkos::parallel_for("compute_evaporation", Kokkos::RangePolicy<ExecutionSpace>(0, nchannels), KOKKOS_LAMBDA(const size_t ij) {
        double D_h = hydraulic_diameter_view(ij, k); // hydraulic diameter [m]
        double P_H = geom->heated_perimeter(ij, k); // heated perimeter [m]
        double A_f = flow_area_view(ij, k); // flow area [m^2]

        double Re = (W_l(ij, k) / A_f) * D_h / mu(ij, k); // Reynolds number
        double Pr = Cp(ij, k) * mu(ij, k) / cond(ij, k); // Prandtl number
        double Pe = Re * Pr; // Peclet number
        double Qflux_wall = lhr(ij, k_node) / P_H; // wall heat flux [W/m^2]
        double G_l = W_l(ij, k) / A_f; // liquid mass flux [kg/m^2-s]
        double G_v = W_v(ij, k) / A_f; // vapor mass flux [kg/m^2-s]
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

            double epsilon = rho(ij, k) * (h_f - h_l(ij, k)) / (rho_g * h_fg); // pumping parameter (Eq. 53)
            double gamma_cond = (H_0 * (1.0 / v_fg) * A_f * alpha(ij, k) * (Tsat - T(ij, k))) / P_H; // condensation rate (Eq. 54)
            evap(ij, k_node) = P_H * Qflux_boil / (h_fg * (1.0 + epsilon)) - P_H * gamma_cond; // Eq. 50

        } else {
            // Saturated region
            Qflux_boil = Qflux_wall;
            evap(ij, k_node) = P_H * Qflux_boil / h_fg;
        }

    });

    Kokkos::fence();
}

template <typename ExecutionSpace>
void TH::solve_mixing(State<ExecutionSpace>& state) {

    const double Thetam = 5.0; // constant set equal to 5.0 for BWR applications, from ANTS Theory

    size_t k = state.surface_plane;  // closure relations use lagging edge values
    size_t k_node = state.node_plane;

    typename State<ExecutionSpace>::View2D spv = state.fluid->mu(state.h_l);
    typename State<ExecutionSpace>::View2D rho_l = state.fluid->rho(state.h_l);

    double rho_f = state.fluid->rho_f();
    double rho_g = state.fluid->rho_g();
    double S_ij = state.geom->gap_width();
    double mu_g = state.fluid->mu_g();
    double mu_f = state.fluid->mu_f();
    double h_g = state.fluid->h_g();

    // Allocate device views for per-channel properties
    typename State<ExecutionSpace>::View1D gbar0("gbar0", state.geom->nchannels());
    typename State<ExecutionSpace>::View1D reyn0("reyn0", state.geom->nchannels());
    typename State<ExecutionSpace>::View1D Theta("Theta", state.geom->nchannels());

    // Capture variables for device lambda
    auto W_l = state.W_l;
    auto W_v = state.W_v;
    auto X = state.X;
    auto alpha = state.alpha;
    auto h_l = state.h_l;
    auto channel_area = state.geom->channel_area_view();
    auto hydraulic_diameter = state.geom->hydraulic_diameter_view();

    // Compute per-channel properties on device
    Kokkos::parallel_for("compute_mixing_properties", Kokkos::RangePolicy<ExecutionSpace>(0, state.geom->nchannels()),
        KOKKOS_LAMBDA(const size_t ij) {
            double A_f = channel_area(ij, k);
            if (A_f < 1e-12) {
                gbar0(ij) = 0.0;
                reyn0(ij) = 0.0;
                Theta(ij) = 1.0;
                return;
            }

            double D_h = hydraulic_diameter(ij, k);
            double viscmi = X(ij, k) / mu_g + (1.0 - X(ij, k)) / mu_f;
            gbar0(ij) = (W_l(ij, k) + W_v(ij, k)) / A_f;

            // Protect against zero flow
            if (gbar0(ij) < 1e-6) {
                reyn0(ij) = 0.0;
                Theta(ij) = 1.0;
                return;
            }

            reyn0(ij) = gbar0(ij) * D_h / viscmi;

            double Xmm = (0.4 * Kokkos::sqrt(rho_f * (rho_f - rho_g) * g * D_h) / gbar0(ij) + 0.6) / (Kokkos::sqrt(rho_f / rho_g) + 0.6);
            double X0m = 0.57 * Kokkos::pow(reyn0(ij), 0.0417);
            double Xfm = X(ij, k) / Xmm;
            if (X(ij, k) < Xmm) {
                Theta(ij) = 1.0 + (Thetam - 1.0) * Xfm;
            } else {
                Theta(ij) = 1.0 + (Thetam - 1.0) * (1.0 - X0m) / (Xfm - X0m);
            }
        });

    Kokkos::fence(); // Ensure gbar0, reyn0, Theta are computed before using them

    // Capture more variables for device lambda
    auto surfaces = state.geom->surfaces;
    auto G_l_tm = state.G_l_tm;
    auto G_v_tm = state.G_v_tm;
    auto Q_m_tm = state.Q_m_tm;
    auto M_m_tm = state.M_m_tm;
    auto G_l_vd = state.G_l_vd;
    auto G_v_vd = state.G_v_vd;
    auto Q_m_vd = state.Q_m_vd;
    auto M_m_vd = state.M_m_vd;

    // Compute mixing terms per surface on device
    Kokkos::parallel_for("compute_mixing_terms", Kokkos::RangePolicy<ExecutionSpace>(0, state.geom->nsurfaces()),
        KOKKOS_LAMBDA(const size_t s) {
            Surface surf = surfaces(s);
            size_t ns = surf.idx;
            size_t i = surf.from_node;
            size_t j = surf.to_node;

            double A_f_i = channel_area(i, k);
            double A_f_j = channel_area(j, k);
            double D_h_i = hydraulic_diameter(i, k);
            double D_h_j = hydraulic_diameter(j, k);

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
            double rho_l_i = rho_l(i, k);
            double rho_l_j = rho_l(j, k);
            double h_l_i = h_l(i, k);
            double h_l_j = h_l(j, k);
            double alpha_i = alpha(i, k);
            double alpha_j = alpha(j, k);

            // Liquid velocity inline calculation
            double V_l_i = (alpha_i < 1.0) ? W_l(i, k) / (A_f_i * (1.0 - alpha_i) * rho_l_i) : 0.0;
            double V_l_j = (alpha_j < 1.0) ? W_l(j, k) / (A_f_j * (1.0 - alpha_j) * rho_l_j) : 0.0;

            // Vapor velocity inline calculation
            double V_v_i = (alpha_i > 0.0) ? W_v(i, k) / (A_f_i * alpha_i * rho_g) : 0.0;
            double V_v_j = (alpha_j > 0.0) ? W_v(j, k) / (A_f_j * alpha_j * rho_g) : 0.0;

            // Quality average with protection against division by zero
            const double K_M = 1.4;
            double G_m_sum = G_m_i + G_m_j;
            double X_bar = 0.0;
            if (G_m_sum > 1e-6) {
                X_bar = K_M * (G_m_i - G_m_j) / G_m_sum;
            }

            double tp_mult = 0.5 * (Theta(i) + Theta(j));
            double lambda = 0.0058 * S_ij / D_rod_i;
            double reynbar = 0.5 * (reyn0(i) + reyn0(j));

            double spbar = 0.5 * (spv(i, k) + spv(j, k));
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
        });

    Kokkos::fence();
}

template <typename ExecutionSpace>
void TH::solve_surface_mass_flux(State<ExecutionSpace>& state) {

    const size_t nchan = state.geom->nchan() * state.geom->nchan();
    const size_t nsurf = state.geom->nsurfaces();
    const size_t k = state.surface_plane;
    const size_t k_node = state.node_plane;
    const double K_ns = 0.5; // gap loss coefficient
    const double gtol = 1e-3; // mass flux perturbation amount
    const double tol = 1e-8; // convergence tolerance
    const double dz = state.geom->dz(k_node); // variable axial spacing
    const double S_ij = state.geom->gap_width();
    const double aspect = state.geom->aspect_ratio();

    // Copy previous plane solution as starting guess for gk
    for (size_t ns = 0; ns < nsurf; ++ns) {
        state.gk(ns, k_node) = state.gk(ns, k_node - 1);
    }

    // Create host mirrors for accessing data
    auto h_P = Kokkos::create_mirror_view(state.P);
    auto h_X = Kokkos::create_mirror_view(state.X);
    auto h_gk = Kokkos::create_mirror_view(state.gk);

    // Copy the updated gk (with previous plane values) to host
    Kokkos::deep_copy(h_gk, state.gk);

    // outer loop for newton iteration convergence
    for (size_t outer_iter = 0; outer_iter < state.max_outer_iter; ++outer_iter) {

        // Residual vectors and Jacobian Matrix as Kokkos Views
        Kokkos::View<double*, ExecutionSpace> f0("f0", nsurf);
        Kokkos::View<double*, ExecutionSpace> f3("f3", nsurf);
        Kokkos::View<double**, ExecutionSpace> dfdg("dfdg", nsurf, nsurf);

        // PLANAR solve
        planar(state);

        // Copy updated data
        Kokkos::deep_copy(h_P, state.P);
        Kokkos::deep_copy(h_X, state.X);
        Kokkos::deep_copy(h_gk, state.gk);

        // calculate the residual vector f0
        for (size_t ns = 0; ns < nsurf; ++ns) {
            size_t i = state.geom->surfaces[ns].from_node;
            size_t j = state.geom->surfaces[ns].to_node;
            size_t i_donor = (h_gk(ns, k_node) >= 0) ? i : j;

            double rho_m = state.fluid->rho_m(h_X(i_donor, k));
            double deltaP = h_P(i, k) - h_P(j, k); // Eq. 56 from ANTS Theory
            double Fns = 0.5 * K_ns * h_gk(ns, k_node) * std::abs(h_gk(ns, k_node)) / rho_m; // Eq. 57 from ANTS Theory
            f0(ns) = -dz * aspect * (deltaP - Fns); // Eq. 55 from ANTS Theory
        }

        // calculate max residual
        double max_res = 0.0;
        for (size_t ns = 0; ns < nsurf; ++ns) {
            max_res = std::max(max_res, std::abs(f0(ns)));
        }

        if (max_res < tol) {
            std::cout << "Converged plane " << k << " in " << outer_iter + 1 << " iterations." << std::endl;
            break;
        }

        for (size_t ns1 = 0; ns1 < nsurf; ++ns1) {

            State perturb_state = state; // reset state to reference prior to perturbation

            // Get host mirror for perturbation
            auto h_perturb_gk = Kokkos::create_mirror_view(perturb_state.gk);
            Kokkos::deep_copy(h_perturb_gk, perturb_state.gk);

            // perturb the mass flux at surface ns1
            if (h_perturb_gk(ns1, k_node) >= 0) h_perturb_gk(ns1, k_node) -= gtol;
            else h_perturb_gk(ns1, k_node) += gtol;

            Kokkos::deep_copy(perturb_state.gk, h_perturb_gk);

            // PLANAR_PERTURB solve
            planar(perturb_state);

            // Copy perturbed results
            auto h_perturb_P = Kokkos::create_mirror_view(perturb_state.P);
            auto h_perturb_X = Kokkos::create_mirror_view(perturb_state.X);
            Kokkos::deep_copy(h_perturb_P, perturb_state.P);
            Kokkos::deep_copy(h_perturb_X, perturb_state.X);
            Kokkos::deep_copy(h_perturb_gk, perturb_state.gk);

            for (size_t ns = 0; ns < nsurf; ++ns) {
                size_t i = state.geom->surfaces[ns].from_node;
                size_t j = state.geom->surfaces[ns].to_node;
                size_t i_donor = (h_perturb_gk(ns, k_node) >= 0) ? i : j;

                double rho_m = perturb_state.fluid->rho_m(h_perturb_X(i_donor, k));
                double deltaP = h_perturb_P(i, k) - h_perturb_P(j, k);
                double Fns = 0.5 * K_ns * h_perturb_gk(ns, k_node) * std::abs(h_perturb_gk(ns, k_node)) / rho_m;
                f3(ns) = -dz * aspect * (deltaP - Fns);

                dfdg(ns, ns1) = (f3(ns) - f0(ns)) / (h_perturb_gk(ns1, k_node) - h_gk(ns1, k_node));
            }
        }

        // solve the system of equations (overwrites f0 as solution vector)
        solve_linear_system(nsurf, dfdg, f0);

        // update mass fluxes from solution
        for (size_t ns = 0; ns < nsurf; ++ns) {
            h_gk(ns, k_node) -= f0(ns);
        }

        Kokkos::deep_copy(state.gk, h_gk);

    } // end outer iteration loop
}

template <typename ExecutionSpace>
void TH::solve_flow_rates(State<ExecutionSpace>& state) {
    // Perform calculations for each surface axial plane
    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;

    // Device-based implementation: accumulate source terms, then update flow rates
    auto W_l = state.W_l;
    auto W_v = state.W_v;
    auto X = state.X;
    auto evap = state.evap;
    auto gk = state.gk;
    auto G_l_tm = state.G_l_tm;
    auto G_v_tm = state.G_v_tm;
    auto G_l_vd = state.G_l_vd;
    auto G_v_vd = state.G_v_vd;
    auto surface_view = state.geom->surface_view();
    auto flow_area_view = state.geom->channel_area_view();
    const double gap_width = state.geom->gap_width();
    const double dz = state.geom->dz(k_node);
    const size_t nsurfaces = state.geom->nsurfaces();
    const size_t nchannels = state.geom->nchannels();

    // Create source term views on device
    Kokkos::View<double*, ExecutionSpace> SS_l("SS_l", nchannels);
    Kokkos::View<double*, ExecutionSpace> SS_v("SS_v", nchannels);

    // Initialize to zero
    Kokkos::deep_copy(SS_l, 0.0);
    Kokkos::deep_copy(SS_v, 0.0);

    // Accumulate source terms from transverse surfaces
    Kokkos::parallel_for("accumulate_flow_sources", Kokkos::RangePolicy<ExecutionSpace>(0, nsurfaces), KOKKOS_LAMBDA(const size_t s) {
        auto surf = surface_view(s);
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;
        size_t i_donor = (gk(ns, k_node) >= 0) ? i : j;

        double sl = gk(ns, k_node) * (1.0 - X(i_donor, k-1)) + G_l_tm(ns) + G_l_vd(ns);
        double sl_term = gap_width * sl;
        Kokkos::atomic_add(&SS_l(i), sl_term);
        Kokkos::atomic_add(&SS_l(j), -sl_term);

        double sv = gk(ns, k_node) * X(i_donor, k-1) + G_v_tm(ns) + G_v_vd(ns);
        double sv_term = gap_width * sv;
        Kokkos::atomic_add(&SS_v(i), sv_term);
        Kokkos::atomic_add(&SS_v(j), -sv_term);
    });

    // Update flow rates on device
    Kokkos::parallel_for("update_flow_rates", Kokkos::RangePolicy<ExecutionSpace>(0, nchannels), KOKKOS_LAMBDA(const size_t ij) {
        double A_f_ijk = flow_area_view(ij, k_node);
        if (A_f_ijk < 1e-12) return; // skip channels with no flow area

        // Update liquid flow rate (Eq. 61 from ANTS Theory)
        W_l(ij, k) = W_l(ij, k-1) - dz * (evap(ij, k_node) + SS_l(ij));
        // W_l(ij, k) = (W_l(ij, k) > 0.0) ? W_l(ij, k) : 0.0; // prevent negative

        // Update vapor flow rate (Eq. 62 from ANTS Theory)
        W_v(ij, k) = W_v(ij, k-1) + dz * (evap(ij, k_node) - SS_v(ij));
        // W_v(ij, k) = (W_v(ij, k) > 0.0) ? W_v(ij, k) : 0.0; // prevent negative

    });

    Kokkos::fence(); // ensure completion
}

template <typename ExecutionSpace>
void TH::solve_enthalpy(State<ExecutionSpace>& state) {
    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;

    // Device-based implementation
    auto h_l = state.h_l;
    auto W_l = state.W_l;
    auto W_v = state.W_v;
    auto lhr = state.lhr;
    auto gk = state.gk;
    auto Q_m_tm = state.Q_m_tm;
    auto Q_m_vd = state.Q_m_vd;
    auto surface_view = state.geom->surface_view();
    auto flow_area_view = state.geom->channel_area_view();
    const double gap_width = state.geom->gap_width();
    const double dz = state.geom->dz(k_node);
    const size_t nsurfaces = state.geom->nsurfaces();
    const size_t nchannels = state.geom->nchannels();
    const double h_g = state.fluid->h_g();

    // Create source term view on device
    Kokkos::View<double*, ExecutionSpace> SS_m("SS_m", nchannels);
    Kokkos::deep_copy(SS_m, 0.0);

    // Accumulate enthalpy source terms from transverse surfaces
    Kokkos::parallel_for("accumulate_enthalpy_sources", Kokkos::RangePolicy<ExecutionSpace>(0, nsurfaces), KOKKOS_LAMBDA(const size_t s) {
        auto surf = surface_view(s);
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;
        size_t i_donor = (gk(ns, k_node) >= 0) ? i : j;

        double h_l_donor = h_l(i_donor, k-1);
        double term = gap_width * (gk(ns, k_node) * h_l_donor + Q_m_tm(ns) + Q_m_vd(ns));

        // Kokkos::printf("Surface %d between channels %d and %d: gk = %f, h_l_donor = %f, term = %f\n", ns, i, j, gk(ns, k_node), h_l_donor, term);

        Kokkos::atomic_add(&SS_m(i), term);
        Kokkos::atomic_add(&SS_m(j), -term);
    });

    // Update liquid enthalpy on device
    Kokkos::parallel_for("update_enthalpy", Kokkos::RangePolicy<ExecutionSpace>(0, nchannels), KOKKOS_LAMBDA(const size_t ij) {
        double A_f = flow_area_view(ij, k_node);
        if (A_f < 1e-12) return; // skip channels with no flow area

        // Eq. 63 from ANTS Theory
        h_l(ij, k) = (
            (W_v(ij, k-1) - W_v(ij, k)) * h_g
            + W_l(ij, k-1) * h_l(ij, k-1) + dz * lhr(ij, k_node)
            - dz * SS_m(ij)
        ) / W_l(ij, k);
    });

    Kokkos::fence();
}

template <typename ExecutionSpace>
void TH::solve_void_fraction(State<ExecutionSpace>& state) {
    const double tol = 1e-6;
    const double eps = 1e-12; // small number to prevent division by zero

    // based on the Chexal-Lellouche drift flux model
    double P = state.P(0, 0); // assuming constant pressure for simplicity

    auto h_alpha = Kokkos::create_mirror_view(state.alpha);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_h_l = Kokkos::create_mirror_view(state.h_l);
    auto h_X = Kokkos::create_mirror_view(state.X);

    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_h_l, state.h_l);
    Kokkos::deep_copy(h_X, state.X);

    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double A = state.geom->flow_area(ij, k);
        double D_h = state.geom->hydraulic_diameter(ij, k);
        if (A < 1e-12) continue; // skip channels with no flow area
        double Gv = h_W_v(ij, k) / A; // vapor mass flux
        double Gl = h_W_l(ij, k) / A; // liquid mass flux

        if (Gv < eps) {
            h_alpha(ij, k) = 0.0;
            continue;
        }

        double h_l = h_h_l(ij, k);
        double h_v = state.fluid->h_f() + h_X(ij, k) * state.fluid->h_fg();
        double rho_g = state.fluid->rho_g();
        double rho_f = state.fluid->rho_f();
        double rho_l = state.fluid->rho(h_l);
        double mu_v = state.fluid->mu(h_v);
        double mu_l = state.fluid->mu(h_l);
        double sigma = state.fluid->sigma();

        double Re_g = __Reynolds(h_W_v(ij, k) / A, D_h, mu_v); // local vapor Reynolds number
        double Re_f = __Reynolds(h_W_l(ij, k) / A, D_h, mu_l); // local liquid Reynolds number
        double Re;
        if (Re_g > Re_f) {
            Re = Re_g;
        } else {
            Re = Re_f;
        }
        double A1 = 1 / (1 + exp(-Re / 60000));
        double B1 = std::min(0.8, A1); // from Zuber correlation
        double B2 = 1.41;

        auto f = [B1, B2, P, rho_g, rho_f, rho_l, sigma, Gv, Gl] (double alpha) {

            // calculate distribution parameter, C_0
            double C1 = 4.0 * P_crit * P_crit / (P * (P_crit - P)); // Eq. 24 from ANTS Theory
            double L = (1.0 - std::exp(-C1 * alpha)) / (1.0 - std::exp(-C1)); // Eq. 23 from ANTS Theory
            double K0 = B1 + (1 - B1) * pow(rho_g / rho_f, 0.25); // Eq. 25 from ANTS Theory
            double r = (1 + 1.57 * (rho_g / rho_f)) / (1 - B1); // Eq. 26 from ANTS Theory
            double C0 = L / (K0 + (1 - K0) * pow(alpha, r)); // Eq. 22 from ANTS Theory

            // calculate drift velocity, V_gj
            double Vgj0 = B2 * pow(((rho_f - rho_g) * g * sigma) / (rho_f * rho_f), 0.25); // Eq. 28 from ANTS Theory
            double Vgj = Vgj0 * pow(1.0 - alpha, B1); // Eq. 27 from ANTS Theory

            return (alpha * C0 - 1.0) * Gv + alpha * C0 * (rho_g / rho_l) * Gl + alpha * rho_g * Vgj;
        };

        auto bisection = [](auto f, double a, double b, double tol = 1e-8, int max_iter = 100) {
            double fa = f(a), fb = f(b);

            if (fa == 0.0) return a;
            if (fb == 0.0) return b;
            if (fa * fb > 0) throw std::runtime_error("Root not bracketed//");

            for (int i = 0; i < max_iter; i++) {
                double c = 0.5 * (a + b);
                double fc = f(c);

                if (std::fabs(fc) < tol || (b - a) < tol) return c;
                if (fa * fc < 0) {
                    b = c;
                    fb = fc;
                } else {
                    a = c;
                    fa = fc;
                }
            }
            return 0.5 * (a + b);
        };

        h_alpha(ij, k) = bisection(f, 0.0, 1.0, tol, state.max_inner_iter); // solve for void fraction using bisection method
    }

    Kokkos::deep_copy(state.alpha, h_alpha);
}

template <typename ExecutionSpace>
void TH::solve_quality(State<ExecutionSpace>& state) {
    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;

    // Device-based implementation
    auto X = state.X;
    auto W_v = state.W_v;
    auto W_l = state.W_l;
    auto flow_area_view = state.geom->channel_area_view();
    const size_t nchannels = state.geom->nchannels();

    Kokkos::parallel_for("compute_quality", Kokkos::RangePolicy<ExecutionSpace>(0, nchannels), KOKKOS_LAMBDA(const size_t ij) {
        double A_f_ijk = flow_area_view(ij, k_node);
        if (A_f_ijk < 1e-12) return; // skip channels with no flow area

        double G_v = W_v(ij, k) / A_f_ijk; // vapor mass flux (Eq. 8)
        double G_l = W_l(ij, k) / A_f_ijk; // liquid mass flux (Eq. 9)
        X(ij, k) = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory
    });

    Kokkos::fence();
}

template <typename ExecutionSpace>
void TH::solve_pressure(State<ExecutionSpace>& state) {

    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    typename State<ExecutionSpace>::View2D rho = state.fluid->rho(state.h_l);
    typename State<ExecutionSpace>::View2D mu = state.fluid->mu(state.h_l);

    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    double dz = state.geom->dz(k_node);

    // Allocate device views for momentum exchange terms
    typename State<ExecutionSpace>::View1D CF_SS("CF_SS", state.geom->nchannels());
    typename State<ExecutionSpace>::View1D TM_SS("TM_SS", state.geom->nchannels());
    typename State<ExecutionSpace>::View1D VD_SS("VD_SS", state.geom->nchannels());

    // Initialize to zero (critical - uninitialized memory causes NaN)
    Kokkos::deep_copy(CF_SS, 0.0);
    Kokkos::deep_copy(TM_SS, 0.0);
    Kokkos::deep_copy(VD_SS, 0.0);

    // Capture variables for device lambda - use raw copies, not shared_ptr
    auto surfaces = state.geom->surfaces;
    auto gk = state.gk;
    auto M_m_tm = state.M_m_tm;
    auto M_m_vd = state.M_m_vd;
    auto W_l = state.W_l;
    auto W_v = state.W_v;
    auto X = state.X;
    auto alpha = state.alpha;
    auto h_l = state.h_l;
    auto channel_area = state.geom->channel_area_view();
    double gap_W = state.geom->gap_width();
    double rho_f = state.fluid->rho_f();
    double rho_g = state.fluid->rho_g();

    // Accumulate surface momentum exchange terms on device
    Kokkos::parallel_for("accumulate_momentum_exchange", Kokkos::RangePolicy<ExecutionSpace>(0, state.geom->nsurfaces()),
        KOKKOS_LAMBDA(const size_t s) {
            Surface surf = surfaces(s);
            size_t ns = surf.idx;
            size_t i = surf.from_node;
            size_t j = surf.to_node;
            size_t i_donor = (gk(ns, k_node) >= 0) ? i : j;

            double A_f_i = channel_area(i, k);
            double A_f_j = channel_area(j, k);
            double A_f_donor = channel_area(i_donor, k-1);

            // Compute V_m for donor channel inline
            double v_m_donor;
            if (alpha(i_donor, k-1) < 1e-6) {
                v_m_donor = 1.0 / rho_f;
            } else if (alpha(i_donor, k-1) > 1.0 - 1e-6) {
                v_m_donor = 1.0 / rho_g;
            } else {
                double X_donor = X(i_donor, k-1);
                double rho_l_donor = rho(i_donor, k-1);
                v_m_donor = (1.0 - X_donor) * (1.0 - X_donor) / ((1.0 - alpha(i_donor, k-1)) * rho_l_donor) +
                            X_donor * X_donor / (alpha(i_donor, k-1) * rho_g);
            }
            double W_m_donor = W_l(i_donor, k-1) + W_v(i_donor, k-1);
            double V_m_donor = v_m_donor * W_m_donor / A_f_donor;

            double cf_term = gap_W * gk(ns, k_node) * V_m_donor;
            Kokkos::atomic_add(&CF_SS(i), cf_term);
            Kokkos::atomic_add(&CF_SS(j), -cf_term);

            double tm_term = gap_W * M_m_tm(ns);
            Kokkos::atomic_add(&TM_SS(i), tm_term);
            Kokkos::atomic_add(&TM_SS(j), -tm_term);

            double vd_term = gap_W * M_m_vd(ns);
            Kokkos::atomic_add(&VD_SS(i), vd_term);
            Kokkos::atomic_add(&VD_SS(j), -vd_term);
        });

    // Synchronize to ensure momentum exchange terms are computed before use
    Kokkos::fence();

    // Compute pressure drops on device
    auto P = state.P;
    auto hydraulic_diameter = state.geom->hydraulic_diameter_view();
    double mu_f = state.fluid->mu_f();
    double mu_g = state.fluid->mu_g();

    Kokkos::parallel_for("compute_pressure", Kokkos::RangePolicy<ExecutionSpace>(0, state.geom->nchannels()),
        KOKKOS_LAMBDA(const size_t ij) {
            double D_h = hydraulic_diameter(ij, k);
            double A_f = channel_area(ij, k);

            if (A_f < 1e-12) return;

            // mass flux (liq. only)
            double G_l = W_l(ij, k) / A_f;

            // mass flux (mixture)
            double G = (W_l(ij, k) + W_v(ij, k)) / A_f;

            // ----- two-phase acceleration pressure drop -----
            // Compute nu_m (specific volume) at k and k-1
            double nu_m_k, nu_m_km1;
            if (alpha(ij, k) < 1e-6) {
                nu_m_k = 1.0 / rho_f;
            } else if (alpha(ij, k) > 1.0 - 1e-6) {
                nu_m_k = 1.0 / rho_g;
            } else {
                nu_m_k = (1.0 - X(ij, k)) * (1.0 - X(ij, k)) / ((1.0 - alpha(ij, k)) * rho(ij, k)) +
                         X(ij, k) * X(ij, k) / (alpha(ij, k) * rho_g);
            }

            if (alpha(ij, k-1) < 1e-6) {
                nu_m_km1 = 1.0 / rho_f;
            } else if (alpha(ij, k-1) > 1.0 - 1e-6) {
                nu_m_km1 = 1.0 / rho_g;
            } else {
                nu_m_km1 = (1.0 - X(ij, k-1)) * (1.0 - X(ij, k-1)) / ((1.0 - alpha(ij, k-1)) * rho(ij, k-1)) +
                           X(ij, k-1) * X(ij, k-1) / (alpha(ij, k-1) * rho_g);
            }

            double dP_accel = G * G * (nu_m_k - nu_m_km1);

            // ----- two-phase frictional pressure drop -----
            // Reynolds number (liq. only)
            double Re = G_l * D_h / mu(ij, k);

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

            // two-phase multiplier for form losses (homogeneous)
            double phi2_hom = 1.0 + X(ij, k) * (rho_f / rho_g - 1.0);

            // two-phase geometry form loss pressure drop
            double dP_form = K_loss * G * G / (2.0 * rho_f) * phi2_hom;

            // two-phase frictional pressure drop
            double dP_tpfric = dP_wall_shear + dP_form;

            // ----- two-phase gravitational pressure drop -----
            double dP_grav = rho(ij, k) * 9.81 * dz;

            // ----- momentum exchange -----
            double dP_CF = dz * CF_SS(ij);
            double dP_TM = dz * TM_SS(ij);
            double dP_VD = dz * VD_SS(ij);
            double dP_momexch = dP_CF + dP_TM + dP_VD;

            // ----- total pressure drop -----
            double dP_total = dP_accel + dP_tpfric + dP_grav + dP_momexch;
            P(ij, k) = P(ij, k-1) - dP_total;
        });

    // Ensure all pressure writes complete
    Kokkos::fence();
}

double TH::__Reynolds(double G, double D_h, double mu) {
    return G * D_h / mu;
}

double TH::__Prandtl(double Cp, double mu, double k) {
    return Cp * mu / k;
}

double TH::__Peclet(double Re, double Pr) {
    return Re * Pr;
}

double TH::__liquid_velocity(double W_l, double A_f, double alpha, double rho_l) {
    if (alpha < 1.0) {
        return W_l / (A_f * (1.0 - alpha) * rho_l);
    }
    return 0.0;
}

double TH::__vapor_velocity(double W_v, double A_f, double alpha, double rho_g) {
    if (alpha > 0.0) {
        return W_v / (A_f * alpha * rho_g);
    }
    return 0.0;
}

double TH::__eddy_velocity(double Re, double S_ij, double D_H_i, double D_H_j, double D_rod, double G_m_i, double rho_m) {
    double lambda = 0.0058 * (S_ij / D_rod); // Eq. 46 from ANTS Theory
    return 0.5 * lambda * pow(Re, -0.1) * (1.0 + pow(D_H_j / D_H_i, 1.5)) * D_H_i / D_rod * G_m_i / rho_m; // Eq. 45 from ANTS Theory
}

double TH::__quality_avg(double G_m_i, double G_m_j) {
    double K_M = 1.4; // constant from ANTS Theory, referenced from Lahey and Moody (1977)
    return K_M * (G_m_i - G_m_j) / (G_m_i + G_m_j); // Eq. 49 from ANTS Theory
}

// Explicit template instantiations
namespace TH {
template void planar<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_evaporation_term<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_mixing<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_surface_mass_flux<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_flow_rates<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_enthalpy<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_void_fraction<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_quality<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);
template void solve_pressure<Kokkos::DefaultExecutionSpace>(State<Kokkos::DefaultExecutionSpace>&);

template void planar<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_evaporation_term<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_mixing<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_surface_mass_flux<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_flow_rates<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_enthalpy<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_void_fraction<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_quality<Kokkos::Serial>(State<Kokkos::Serial>&);
template void solve_pressure<Kokkos::Serial>(State<Kokkos::Serial>&);


}
