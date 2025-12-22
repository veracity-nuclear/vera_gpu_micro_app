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
    double D_h = state.geom->hydraulic_diameter(); // hydraulic diameter [m]
    double P_H = state.geom->heated_perimeter(); // heated perimeter [m]
    double A_f = state.geom->flow_area(); // average flow area [m^2] for single assembly

    typename State<ExecutionSpace>::View2D mu = state.fluid->mu(state.h_l); // dynamic viscosity [Pa-s]
    typename State<ExecutionSpace>::View2D rho = state.fluid->rho(state.h_l); // liquid density [kg/m^3]
    typename State<ExecutionSpace>::View2D cond = state.fluid->k(state.h_l); // thermal conductivity [W/m-K]
    typename State<ExecutionSpace>::View2D Cp = state.fluid->Cp(state.h_l); // specific heat [J/kg-K]
    typename State<ExecutionSpace>::View2D T = state.fluid->T(state.h_l); // temperature [K]

    // Create host mirrors for computation
    auto h_mu = Kokkos::create_mirror_view(mu);
    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_cond = Kokkos::create_mirror_view(cond);
    auto h_Cp = Kokkos::create_mirror_view(Cp);
    auto h_T = Kokkos::create_mirror_view(T);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_h_l = Kokkos::create_mirror_view(state.h_l);
    auto h_alpha = Kokkos::create_mirror_view(state.alpha);
    auto h_lhr = Kokkos::create_mirror_view(state.lhr);
    auto h_evap = Kokkos::create_mirror_view(state.evap);

    Kokkos::deep_copy(h_mu, mu);
    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_cond, cond);
    Kokkos::deep_copy(h_Cp, Cp);
    Kokkos::deep_copy(h_T, T);
    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_h_l, state.h_l);
    Kokkos::deep_copy(h_alpha, state.alpha);
    Kokkos::deep_copy(h_lhr, state.lhr);

    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double Re = __Reynolds(h_W_l(ij, k) / A_f, D_h, h_mu(ij, k)); // Reynolds number
        double Pr = __Prandtl(h_Cp(ij, k), h_mu(ij, k), h_cond(ij, k)); // Prandtl number
        double Pe = __Peclet(Re, Pr); // Peclet number
        double Qflux_wall = h_lhr(ij, k_node) / P_H; // wall heat flux [W/m^2]
        double G_l = h_W_l(ij, k) / A_f; // liquid mass flux [kg/m^2-s]
        double G_v = h_W_v(ij, k) / A_f; // vapor mass flux [kg/m^2-s]
        double G_m = G_l + G_v; // mixture mass flux [kg/m^2-s], Eq. 11 from ANTS Theory

        double void_dc; // void departure, Eq. 52 from ANTS Theory
        if (Pe < 70000.) {
            void_dc = 0.0022 * Pe * (Qflux_wall / G_m);
        } else {
            void_dc = 154.0 * (Qflux_wall / G_m);
        }

        double Qflux_boil; // boiling heat flux [W/m], Eq. 51 from ANTS Theory
        if (h_h_l(ij, k) < state.fluid->h_f()) {
            if ((state.fluid->h_f() - h_h_l(ij, k)) < void_dc) {
                Qflux_boil = Qflux_wall * (1 - ((state.fluid->h_f() - h_h_l(ij, k)) / void_dc));
            } else {
                Qflux_boil = 0.0;
            }

            double epsilon = h_rho(ij, k) * (state.fluid->h_f() - h_h_l(ij, k)) / (state.fluid->rho_g() * state.fluid->h_fg()); // pumping parameter, Eq. 53 from ANTS Theory
            double H_0 = 0.075; // [s^-1 K^-1], condensation parameter; value recommended by Lahey and Moody (1996)
            double gamma_cond = (H_0 * (1 / state.fluid->v_fg()) * A_f * h_alpha(ij, k) * (state.fluid->Tsat() - h_T(ij, k))) / P_H; // condensation rate [kg/m^3-s], Eq. 54 from ANTS Theory
            h_evap(ij, k_node) = P_H * Qflux_boil / (state.fluid->h_fg() * (1 + epsilon)) - P_H * gamma_cond; // Eq. 50 from ANTS Theory

        } else {
            Qflux_boil = Qflux_wall;
            h_evap(ij, k_node) = P_H * Qflux_boil / state.fluid->h_fg();
        }
    }

    Kokkos::deep_copy(state.evap, h_evap);
}

template <typename ExecutionSpace>
void TH::solve_mixing(State<ExecutionSpace>& state) {

    const double Thetam = 5.0; // constant set equal to 5.0 for BWR applications, from ANTS Theory

    size_t k = state.surface_plane;  // closure relations use lagging edge values
    size_t k_node = state.node_plane;

    typename State<ExecutionSpace>::View2D rho_l = state.fluid->rho(state.h_l);
    typename State<ExecutionSpace>::View2D spv = state.fluid->mu(state.h_l);

    // Create host mirrors
    auto h_rho_l = Kokkos::create_mirror_view(rho_l);
    auto h_spv = Kokkos::create_mirror_view(spv);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_h_l = Kokkos::create_mirror_view(state.h_l);
    auto h_alpha = Kokkos::create_mirror_view(state.alpha);
    auto h_X = Kokkos::create_mirror_view(state.X);
    auto h_G_l_tm = Kokkos::create_mirror_view(state.G_l_tm);
    auto h_G_v_tm = Kokkos::create_mirror_view(state.G_v_tm);
    auto h_Q_m_tm = Kokkos::create_mirror_view(state.Q_m_tm);
    auto h_M_m_tm = Kokkos::create_mirror_view(state.M_m_tm);
    auto h_G_l_vd = Kokkos::create_mirror_view(state.G_l_vd);
    auto h_G_v_vd = Kokkos::create_mirror_view(state.G_v_vd);
    auto h_Q_m_vd = Kokkos::create_mirror_view(state.Q_m_vd);
    auto h_M_m_vd = Kokkos::create_mirror_view(state.M_m_vd);

    Kokkos::deep_copy(h_rho_l, rho_l);
    Kokkos::deep_copy(h_spv, spv);
    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_h_l, state.h_l);
    Kokkos::deep_copy(h_alpha, state.alpha);
    Kokkos::deep_copy(h_X, state.X);
    Kokkos::deep_copy(state.geom->surfaces, state.geom->surfaces);

    double rhof = state.fluid->rho_f();
    double rho_g = state.fluid->rho_g();

    double A_f = state.geom->flow_area(); // average flow area [m^2] for single assembly
    double D_rod = state.geom->heated_perimeter() / M_PI; // rod diameter [m], assuming square array
    double D_h = state.geom->hydraulic_diameter(); // hydraulic diameter [m]
    double S_ij = state.geom->gap_width(); // gap width between subchannels [m]

    // precalculate the two-phase multipliers on a subchannel basis
    std::vector<double> gbar0(state.geom->nchannels(), 0.0);
    std::vector<double> reyn0(state.geom->nchannels(), 0.0);
    std::vector<double> Theta(state.geom->nchannels(), 0.0);
    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double A_f_ijk = state.geom->flow_area(ij, k_node); // flow area for channel ij at axial node k_node
        if (A_f_ijk < 1e-12) continue; // skip channels with no flow area
        double viscmi = h_X(ij, k) / state.fluid->mu_g() + (1.0 - h_X(ij, k)) / state.fluid->mu_f();
        gbar0[ij] = (h_W_l(ij, k) + h_W_v(ij, k)) / A_f_ijk;
        reyn0[ij] = gbar0[ij] * D_h / viscmi;

        double Xmm = (0.4 * std::sqrt(rhof * (rhof - rho_g) * g * D_h) / gbar0[ij] + 0.6) / (std::sqrt(rhof / rho_g) + 0.6);
        double X0m = 0.57 * std::pow(reyn0[ij], 0.0417);
        double Xfm = h_X(ij, k) / Xmm;
        if (h_X(ij, k) < Xmm) {
            Theta[ij] = 1.0 + (Thetam - 1.0) * Xfm;
        } else {
            Theta[ij] = 1.0 + (Thetam - 1.0) * (1.0 - X0m) / (Xfm - X0m);
        }
    }

    // loop over surfaces
    for (size_t s = 0; s < state.geom->nsurfaces(); ++s) {
        Surface surf = state.geom->surfaces(s);
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;

        double G_m_i = state.W_m(i, k) / A_f;
        double G_m_j = state.W_m(j, k) / A_f;
        double G_m_avg = 0.5 * (G_m_i + G_m_j);
        double rho_l_i = h_rho_l(i, k);
        double rho_l_j = h_rho_l(j, k);
        double h_l_i = h_h_l(i, k);
        double h_l_j = h_h_l(j, k);
        double h_l_avg = 0.5 * (h_l_i + h_l_j);
        double alpha_i = h_alpha(i, k);
        double alpha_j = h_alpha(j, k);
        double V_l_i = __liquid_velocity(h_W_l(i, k), A_f, alpha_i, rho_l_i);
        double V_l_j = __liquid_velocity(h_W_l(j, k), A_f, alpha_j, rho_l_j);
        double V_v_i = __vapor_velocity(h_W_v(i, k), A_f, alpha_i, rho_g);
        double V_v_j = __vapor_velocity(h_W_v(j, k), A_f, alpha_j, rho_g);
        double Re = __Reynolds(G_m_avg, D_h, state.fluid->mu(h_l_avg));
        double X_bar = __quality_avg(G_m_i, G_m_j);
        double tp_mult = 0.5 * (Theta[i] + Theta[j]);
        double lambda = 0.0058 * S_ij / D_rod; // Eq. 46 from ANTS Theory
        double reynbar = 0.5 * (reyn0[i] + reyn0[j]);
        double spbar = 0.5 * (h_spv(i, k) + h_spv(j, k));
        double eddy_V;
        if (reyn0[i] < reyn0[j]) {
            eddy_V= 0.5 * lambda * std::pow(reynbar, -0.1) * (1.0 + std::pow(D_h / D_h, 1.5)) * (D_h / D_rod) * gbar0[i] * spbar;
        } else {
            eddy_V= 0.5 * lambda * std::pow(reynbar, -0.1) * (1.0 + std::pow(D_h / D_h, 1.5)) * (D_h / D_rod) * gbar0[j] * spbar;
        }

        // turbulent mixing liquid mass transfer
        h_G_l_tm(ns) = eddy_V * tp_mult * (
            (1 - alpha_i) * rho_l_i - (1 - alpha_j) * rho_l_j
        ); // Eq. 37 from ANTS Theory

        // turbulent mixing vapor mass transfer
        h_G_v_tm(ns) = eddy_V * tp_mult * state.fluid->rho_g() * (alpha_i - alpha_j); // Eq. 38 from ANTS Theory

        // turbulent mixing energy transfer
        h_Q_m_tm(ns) = eddy_V * tp_mult * (
            (1 - alpha_i) * rho_l_i * h_l_i + alpha_i * state.fluid->rho_g() * state.fluid->h_g()
            - (1 - alpha_j) * rho_l_j * h_l_j - alpha_j * state.fluid->rho_g() * state.fluid->h_g()
        ); // Eq. 39 from ANTS Theory

        // turbulent mixing momentum transfer
        h_M_m_tm(ns) = eddy_V * tp_mult * (G_m_i - G_m_j); // Eq. 40 from ANTS Theory

        // void drift liquid mass transfer
        h_G_l_vd(ns) = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i + alpha_j * rho_l_j
        ); // Eq. 42 from ANTS Theory

        // void drift vapor mass transfer
        h_G_v_vd(ns) = -eddy_V * tp_mult * X_bar * (
            alpha_i + alpha_j
        ) * state.fluid->rho_g(); // Eq. 41 from ANTS Theory

        // void drift energy transfer
        h_Q_m_vd(ns) = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i * h_l_i + alpha_j * rho_l_j * h_l_j
            - (alpha_i + alpha_j) * state.fluid->rho_g() * state.fluid->h_g()
        ); // Eq. 43 from ANTS Theory

        // void drift momentum transfer
        h_M_m_vd(ns) = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i * V_l_i + alpha_j * rho_l_j * V_l_j
            - (alpha_i * V_v_i + alpha_j * V_v_j) * state.fluid->rho_g()
        ); // Eq. 44 from ANTS Theory
    }

    Kokkos::deep_copy(state.G_l_tm, h_G_l_tm);
    Kokkos::deep_copy(state.G_v_tm, h_G_v_tm);
    Kokkos::deep_copy(state.Q_m_tm, h_Q_m_tm);
    Kokkos::deep_copy(state.M_m_tm, h_M_m_tm);
    Kokkos::deep_copy(state.G_l_vd, h_G_l_vd);
    Kokkos::deep_copy(state.G_v_vd, h_G_v_vd);
    Kokkos::deep_copy(state.Q_m_vd, h_Q_m_vd);
    Kokkos::deep_copy(state.M_m_vd, h_M_m_vd);
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

    // Create host mirrors
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_X = Kokkos::create_mirror_view(state.X);
    auto h_evap = Kokkos::create_mirror_view(state.evap);
    auto h_gk = Kokkos::create_mirror_view(state.gk);
    auto h_G_l_tm = Kokkos::create_mirror_view(state.G_l_tm);
    auto h_G_v_tm = Kokkos::create_mirror_view(state.G_v_tm);
    auto h_G_l_vd = Kokkos::create_mirror_view(state.G_l_vd);
    auto h_G_v_vd = Kokkos::create_mirror_view(state.G_v_vd);

    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_X, state.X);
    Kokkos::deep_copy(h_evap, state.evap);
    Kokkos::deep_copy(h_gk, state.gk);
    Kokkos::deep_copy(h_G_l_tm, state.G_l_tm);
    Kokkos::deep_copy(h_G_v_tm, state.G_v_tm);
    Kokkos::deep_copy(h_G_l_vd, state.G_l_vd);
    Kokkos::deep_copy(h_G_v_vd, state.G_v_vd);

    // loop over transverse surfaces to add source terms to flow rates
    std::vector<double> SS_l(state.geom->nchannels());
    std::vector<double> SS_v(state.geom->nchannels());
    for (size_t s = 0; s < state.geom->nsurfaces(); ++s) {
        Surface surf = state.geom->surfaces(s);
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;
        size_t i_donor = (h_gk(ns, k_node) >= 0) ? i : j;

        double sl = h_gk(ns, k_node) * (1.0 - h_X(i_donor, k-1)) + h_G_l_tm(ns) + h_G_l_vd(ns);
        SS_l[i] += state.geom->gap_width() * sl;
        SS_l[j] -= state.geom->gap_width() * sl;

        double sv = h_gk(ns, k_node) * h_X(i_donor, k-1) + h_G_v_tm(ns) + h_G_v_vd(ns);
        SS_v[i] += state.geom->gap_width() * sv;
        SS_v[j] -= state.geom->gap_width() * sv;
    }

    const double dz = state.geom->dz(k_node); // variable axial spacing
    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double A_f_ijk = state.geom->flow_area(ij, k_node);
        if (A_f_ijk < 1e-12) continue; // skip channels with no flow area

        h_W_l(ij, k) = h_W_l(ij, k-1) - dz * (h_evap(ij, k_node) + SS_l[ij]); // Eq. 61 from ANTS Theory
        h_W_l(ij, k) = std::max(h_W_l(ij, k), 0.0); // prevent negative liquid flow rate

        // throw error if liquid flow rate becomes negative and add debug info
        if (h_W_l(ij, k) < 0) {
            throw std::runtime_error("Error: Liquid flow rate has become negative in channel " + std::to_string(ij) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "Previous W_l: " + std::to_string(h_W_l(ij, k-1)) + "\n"
                "Evaporation term: " + std::to_string(h_evap(ij, k_node)) + "\n"
                "SS_l: " + std::to_string(SS_l[ij]) + "\n");
        }

        h_W_v(ij, k) = h_W_v(ij, k-1) + dz * (h_evap(ij, k_node) - SS_v[ij]); // Eq. 62 from ANTS Theory
        h_W_v(ij, k) = std::max(h_W_v(ij, k), 0.0); // prevent negative vapor flow rate

        // throw error if vapor flow rate becomes negative and add debug info
        if (h_W_v(ij, k) < 0) {
            throw std::runtime_error("Error: Vapor flow rate has become negative in channel " + std::to_string(ij) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "Previous W_v: " + std::to_string(h_W_v(ij, k-1)) + "\n"
                "Evaporation term: " + std::to_string(h_evap(ij, k_node)) + "\n"
                "SS_v: " + std::to_string(SS_v[ij]) + "\n");
        }
    }

    Kokkos::deep_copy(state.W_l, h_W_l);
    Kokkos::deep_copy(state.W_v, h_W_v);
}

template <typename ExecutionSpace>
void TH::solve_enthalpy(State<ExecutionSpace>& state) {
    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;

    // Create host mirrors
    auto h_h_l = Kokkos::create_mirror_view(state.h_l);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_lhr = Kokkos::create_mirror_view(state.lhr);
    auto h_gk = Kokkos::create_mirror_view(state.gk);
    auto h_Q_m_tm = Kokkos::create_mirror_view(state.Q_m_tm);
    auto h_Q_m_vd = Kokkos::create_mirror_view(state.Q_m_vd);

    Kokkos::deep_copy(h_h_l, state.h_l);
    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_lhr, state.lhr);
    Kokkos::deep_copy(h_gk, state.gk);
    Kokkos::deep_copy(h_Q_m_tm, state.Q_m_tm);
    Kokkos::deep_copy(h_Q_m_vd, state.Q_m_vd);

    // loop over transverse surfaces to add source terms to mixture enthalpy
    std::vector<double> SS_m(state.geom->nchannels());
    for (size_t s = 0; s < state.geom->nsurfaces(); ++s) {
        Surface surf = state.geom->surfaces(s);
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;
        size_t i_donor = (h_gk(ns, k_node) >= 0) ? i : j;

        double h_l_donor = h_h_l(i_donor, k-1);

        SS_m[i] += state.geom->gap_width() * (h_gk(ns, k_node) * h_l_donor + h_Q_m_tm(ns) + h_Q_m_vd(ns));
        SS_m[j] -= state.geom->gap_width() * (h_gk(ns, k_node) * h_l_donor + h_Q_m_tm(ns) + h_Q_m_vd(ns));
    }

    const double dz = state.geom->dz(k_node); // variable axial spacing
    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double A_f_ijk = state.geom->flow_area(ij, k_node);
        if (A_f_ijk < 1e-12) continue; // skip channels with no flow area

        h_h_l(ij, k) = (
            (h_W_v(ij, k-1) - h_W_v(ij, k)) * state.fluid->h_g()
            + h_W_l(ij, k-1) * h_h_l(ij, k-1) + dz * h_lhr(ij, k_node)
            - dz * SS_m[ij]
        ) / h_W_l(ij, k); // Eq. 63 from ANTS Theory
    }

    Kokkos::deep_copy(state.h_l, h_h_l);
}

template <typename ExecutionSpace>
void TH::solve_void_fraction(State<ExecutionSpace>& state) {
    const double tol = 1e-6;
    const double eps = 1e-12; // small number to prevent division by zero

    // based on the Chexal-Lellouche drift flux model
    double P = state.P(0, 0); // assuming constant pressure for simplicity
    double A = state.geom->flow_area(); // average flow area for single assembly
    double D_h = state.geom->hydraulic_diameter();

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
        double A_ijk = state.geom->flow_area(ij, k_node);
        if (A_ijk < 1e-12) continue; // skip channels with no flow area
        double Gv = h_W_v(ij, k) / A_ijk; // vapor mass flux
        double Gl = h_W_l(ij, k) / A_ijk; // liquid mass flux

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

        double Re_g = __Reynolds(h_W_v(ij, k) / A_ijk, D_h, mu_v); // local vapor Reynolds number
        double Re_f = __Reynolds(h_W_l(ij, k) / A_ijk, D_h, mu_l); // local liquid Reynolds number
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

    auto h_X = Kokkos::create_mirror_view(state.X);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);

    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_W_l, state.W_l);

    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double A_f_ijk = state.geom->flow_area(ij, k_node); // flow area for channel ij at axial node k_node
        if (A_f_ijk < 1e-12) continue; // skip channels with no flow area
        double G_v = h_W_v(ij, k) / A_f_ijk; // vapor mass flux, Eq. 8 from ANTS Theory
        double G_l = h_W_l(ij, k) / A_f_ijk; // liquid mass flux, Eq. 9 from ANTS Theory
        h_X(ij, k) = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory

        // throw error if quality becomes negative and add debug info
        if (h_X(ij, k) < 0.0) {
            throw std::runtime_error("Error: Quality has become negative in channel " + std::to_string(ij) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "G_v: " + std::to_string(G_v) + "\n"
                "G_l: " + std::to_string(G_l) + "\n");
        }
    }

    Kokkos::deep_copy(state.X, h_X);
}

template <typename ExecutionSpace>
void TH::solve_pressure(State<ExecutionSpace>& state) {

    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    double D_h = state.geom->hydraulic_diameter();
    double A_f = state.geom->flow_area(); // average flow area for single assembly

    typename State<ExecutionSpace>::View2D rho = state.fluid->rho(state.h_l);
    typename State<ExecutionSpace>::View2D mu = state.fluid->mu(state.h_l);

    // Create host mirrors
    auto h_rho = Kokkos::create_mirror_view(rho);
    auto h_mu = Kokkos::create_mirror_view(mu);
    auto h_W_l = Kokkos::create_mirror_view(state.W_l);
    auto h_W_v = Kokkos::create_mirror_view(state.W_v);
    auto h_X = Kokkos::create_mirror_view(state.X);
    auto h_P = Kokkos::create_mirror_view(state.P);
    auto h_gk = Kokkos::create_mirror_view(state.gk);
    auto h_M_m_tm = Kokkos::create_mirror_view(state.M_m_tm);
    auto h_M_m_vd = Kokkos::create_mirror_view(state.M_m_vd);

    Kokkos::deep_copy(h_rho, rho);
    Kokkos::deep_copy(h_mu, mu);
    Kokkos::deep_copy(h_W_l, state.W_l);
    Kokkos::deep_copy(h_W_v, state.W_v);
    Kokkos::deep_copy(h_X, state.X);
    Kokkos::deep_copy(h_P, state.P);
    Kokkos::deep_copy(h_gk, state.gk);
    Kokkos::deep_copy(h_M_m_tm, state.M_m_tm);
    Kokkos::deep_copy(h_M_m_vd, state.M_m_vd);

    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    double dz = state.geom->dz(k_node); // variable axial spacing

    // loop over transverse surfaces to add source terms to pressure drops
    std::vector<double> CF_SS(state.geom->nchannels()); // sum of cross-flow momentum exchange terms [Pa]
    std::vector<double> TM_SS(state.geom->nchannels()); // sum of turbulent mixing liquid momentum exchange terms [Pa]
    std::vector<double> VD_SS(state.geom->nchannels()); // sum of void drift liquid momentum exchange terms [Pa]

    for (size_t s = 0; s < state.geom->nsurfaces(); ++s) {
        Surface surf = state.geom->surfaces(s);
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;
        size_t i_donor = (h_gk(ns, k_node) >= 0) ? i : j;

        CF_SS[i] += state.geom->gap_width() * h_gk(ns, k_node) * state.V_m(i_donor, k-1);
        CF_SS[j] -= state.geom->gap_width() * h_gk(ns, k_node) * state.V_m(i_donor, k-1);

        TM_SS[i] += state.geom->gap_width() * h_M_m_tm(ns);
        TM_SS[j] -= state.geom->gap_width() * h_M_m_tm(ns);

        VD_SS[i] += state.geom->gap_width() * h_M_m_vd(ns);
        VD_SS[j] -= state.geom->gap_width() * h_M_m_vd(ns);
    }

    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double A_f_ijk = state.geom->flow_area(ij, k_node);
        if (A_f_ijk < 1e-12) continue; // skip channels with no flow area

        // ----- two-phase acceleration pressure drop -----
        double dP_accel = (state.W_m(ij, k) * state.V_m(ij, k) - state.W_m(ij, k-1) * state.V_m(ij, k-1)) / A_f;

        // ----- two-phase frictional pressure drop -----
        // mass flux (liq. only)
        double G_l = (h_W_l(ij, k)) / A_f;

        // mass flux (mixture)
        double G = (h_W_l(ij, k) + h_W_v(ij, k)) / A_f;

        // Reynolds number (liq. only)
        double Re = __Reynolds(G_l, D_h, h_mu(ij, k));

        // frictional pressure drop from wall shear
        double f = a_1 * pow(Re, n); // single phase friction factor, Eq. 31 from ANTS Theory
        double K = f * dz / D_h; // frictional loss coefficient, Eq. 30 from ANTS Theory
        double gamma = pow(state.fluid->rho_f() / state.fluid->rho_g(), 0.5) * pow(state.fluid->mu_g() / state.fluid->mu_f(), 0.2); // Eq. 33 from ANTS Theory

        // parameter b for two-phase multiplier (Chisholm), Eq. 34 from ANTS Theory
        double b;
        if (gamma <= 9.5) {
            b = 55.0 / sqrt(G);
        } else if (gamma < 28) {
            b = 520.0 / (gamma * sqrt(G));
        } else {
            b = 15000.0 / (gamma * gamma * sqrt(G));
        }

        // two-phase multiplier for wall shear (Chisholm), Eq. 32 from ANTS Theory
        double phi2_ch = 1.0 + (gamma * gamma - 1.0) * (b * pow(h_X(ij, k), 0.9) * pow((1.0 - h_X(ij, k)), 0.9) + pow(h_X(ij, k), 1.8));

        // throw error if phi2_ch is NaN and add debug info
        if (std::isnan(phi2_ch)) {
            throw std::runtime_error("Error: Two-phase multiplier has become NaN in channel " + std::to_string(ij) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "gamma: " + std::to_string(gamma) + "\n"
                "b: " + std::to_string(b) + "\n"
                "X: " + std::to_string(h_X(ij, k)) + "\n");
        }

        // two-phase wall shear pressure drop, Eq. 29 from ANTS Theory
        double dP_wall_shear = K * G * G / (2.0 * state.fluid->rho_f()) * phi2_ch;

        // throw error if dP_wall_shear is NaN and add debug info
        if (std::isnan(dP_wall_shear)) {
            throw std::runtime_error("Error: Two-phase wall shear pressure drop has become NaN in channel " + std::to_string(ij) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "K: " + std::to_string(K) + "\n"
                "G: " + std::to_string(G) + "\n"
                "rho: " + std::to_string(state.fluid->rho_f()) + "\n"
                "phi2_ch: " + std::to_string(phi2_ch) + "\n"
            );
        }

        // form loss coefficient (no form losses in this simple model)
        double K_loss = 0.0;

        // two-phase multiplier for form losses (homogeneous), Eq. 35 from ANTS Theory
        double phi2_hom = 1.0 + h_X(ij, k) * (state.fluid->rho_f() / state.fluid->rho_g() - 1.0);

        // two-phase geometry form loss pressure drop, Eq. 36 from ANTS Theory
        double dP_form = K_loss * G * G / (2.0 * state.fluid->rho_f()) * phi2_hom;

        // two-phase frictional pressure drop, Eq. 36 from ANTS Theory
        double dP_tpfric = dP_wall_shear + dP_form;

        // throw error if dP_tpfric is NaN and add debug info
        if (std::isnan(dP_tpfric)) {
            throw std::runtime_error("Error: Two-phase frictional pressure drop has become NaN in channel " + std::to_string(ij) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "dP_wall_shear: " + std::to_string(dP_wall_shear) + "\n"
                "dP_form: " + std::to_string(dP_form) + "\n");
        }

        // ----- two-phase gravitational pressure drop -----
        double dP_grav = h_rho(ij, k) * g * dz;

        // ----- momentum exchange due to pressure-directed crossflow, turbulent mixing, and void drift -----
        double dP_CF = dz / A_f * CF_SS[ij];
        double dP_TM = dz / A_f * TM_SS[ij];
        double dP_VD = dz / A_f * VD_SS[ij];
        double dP_momexch = dP_CF + dP_TM + dP_VD;

        // ----- total pressure drop over this axial plane -----
        double dP_total = dP_accel + dP_tpfric + dP_grav + dP_momexch; // Eq. 65 from ANTS Theory
        h_P(ij, k) = h_P(ij, k-1) - dP_total;

        // throw error if pressure goes NaN and add debug info (dP_accel, dP_tpfric, dP_grav, dP_momexch, dP_total)
        if (std::isnan(h_P(ij, k))) {
            throw std::runtime_error("Error: Pressure has become NaN in channel " + std::to_string(ij) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "dP_accel: " + std::to_string(dP_accel) + "\n"
                "dP_tpfric: " + std::to_string(dP_tpfric) + "\n"
                "dP_grav: " + std::to_string(dP_grav) + "\n"
                "dP_momexch: " + std::to_string(dP_momexch) + "\n"
                "dP_total: " + std::to_string(dP_total) + "\n");
        }
    }

    Kokkos::deep_copy(state.P, h_P);
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
