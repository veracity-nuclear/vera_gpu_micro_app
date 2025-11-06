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

void TH::solve_evaporation_term(State& state) {
    double D_h = state.geom->hydraulic_diameter(); // hydraulic diameter [m]
    double P_H = state.geom->heated_perimeter(); // heated perimeter [m]
    double A_f = state.geom->flow_area(); // flow area [m^2]

    Vector2D mu = state.fluid->mu(state.h_l); // dynamic viscosity [Pa-s]
    Vector2D rho = state.fluid->rho(state.h_l); // liquid density [kg/m^3]
    Vector2D cond = state.fluid->k(state.h_l); // thermal conductivity [W/m-K]
    Vector2D Cp = state.fluid->Cp(state.h_l); // specific heat [J/kg-K]
    Vector2D T = state.fluid->T(state.h_l); // temperature [K]

    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double Re = __Reynolds(state.W_l[i][k] / A_f, D_h, mu[i][k]); // Reynolds number
        double Pr = __Prandtl(Cp[i][k], mu[i][k], cond[i][k]); // Prandtl number
        double Pe = __Peclet(Re, Pr); // Peclet number
        double Qflux_wall = state.lhr[i][k] / P_H; // wall heat flux [W/m^2]
        double G_l = state.W_l[i][k] / A_f; // liquid mass flux [kg/m^2-s]
        double G_v = state.W_v[i][k] / A_f; // vapor mass flux [kg/m^2-s]
        double G_m = G_l + G_v; // mixture mass flux [kg/m^2-s], Eq. 11 from ANTS Theory

        double void_dc; // void departure, Eq. 52 from ANTS Theory
        if (Pe < 70000.) {
            void_dc = 0.0022 * Pe * (Qflux_wall / G_m);
        } else {
            void_dc = 154.0 * (Qflux_wall / G_m);
        }

        if (state.h_l[i][k] < state.fluid->h_f()) {
            double Qflux_boil; // boiling heat flux [W/m], Eq. 51 from ANTS Theory
            if ((state.fluid->h_f() - state.h_l[i][k]) < void_dc) {
                Qflux_boil = Qflux_wall * (1 - ((state.fluid->h_f() - state.h_l[i][k]) / void_dc));
            } else {
                Qflux_boil = 0.0;
            }

            double epsilon = rho[i][k] * (state.fluid->h_f() - state.h_l[i][k]) / (state.fluid->rho_g() * state.fluid->h_fg()); // pumping parameter, Eq. 53 from ANTS Theory
            double H_0 = 0.075; // [s^-1 K^-1], condensation parameter; value recommended by Lahey and Moody (1996)
            double gamma_cond = (H_0 * (1 / state.fluid->v_fg()) * A_f * state.alpha[i][k] * (state.fluid->Tsat() - T[i][k])) / P_H; // condensation rate [kg/m^3-s], Eq. 54 from ANTS Theory
            state.evap[i][k_node] = P_H * Qflux_boil / (state.fluid->h_fg() * (1 + epsilon)) - P_H * gamma_cond; // Eq. 50 from ANTS Theory

        } else {
            double Qflux_boil = Qflux_wall;
            state.evap[i][k_node] = P_H * Qflux_boil / state.fluid->h_fg();
        }
    }
}

// void TH::solve_turbulent_mixing(State& state) {

//     size_t k = state.surface_plane;  // closure relations use lagging edge values
//     size_t k_node = state.node_plane;

//     Vector2D rho_l = state.fluid->rho(state.h_l);

//     double A_f = state.geom->flow_area(); // flow area [m^2]
//     double D_rod = state.geom->heated_perimeter() / M_PI; // rod diameter [m], assuming square array
//     double D_H_i = state.geom->hydraulic_diameter(); // hydraulic diameter of subchannel i [m]
//     double D_H_j = state.geom->hydraulic_diameter(); // hydraulic diameter of subchannel j [m]
//     double S_ij = state.geom->gap_width(); // gap width between subchannels [m]

//     size_t i_neigh;
//     size_t j_neigh;

//     // loop over subchannel nodes
//     for (size_t i = 0; i < state.geom->nchannels(); ++i) {
//         // loop over 4 surfaces between neighboring subchannels; 0: west, 1: east, 2: north, 3: south
//         for (size_t ns = 0; ns < 4; ++ns) {
//             // perform calculations for surface ns between subchannels i and j
//             size_t i_neigh; size_t j_neigh;
//             std::tie(i_neigh, j_neigh) = __get_neighbor_ij(i, j, ns, state.geom->nx(), state.geom->ny());

//             if ((i_neigh == -1) || (j_neigh == -1)) {
//                 continue; // skip if neighbor is out of bounds
//             }

//             double X = state.X[i][k]; // quality in subchannel i
//             double G_m_i = state.W_m(i, k) / A_f;
//             double G_m_j = state.W_m(i_neigh, k) / A_f;
//             double rho_l_i = rho_l[i][k];
//             double rho_l_j = rho_l[i_neigh][k];
//             double h_l_i = state.h_l[i][k];
//             double h_l_j = state.h_l[i_neigh][k];
//             double alpha_i = state.alpha[i][k];
//             double alpha_j = state.alpha[i_neigh][k];
//             double V_l_i = __liquid_velocity(state.W_l[i][k], A_f, alpha_i, rho_l_i);
//             double V_l_j = __liquid_velocity(state.W_l[i_neigh][k], A_f, alpha_j, rho_l_j);
//             double V_v_i = __vapor_velocity(state.W_v[i][k], A_f, alpha_i, state.fluid->rho_g());
//             double V_v_j = __vapor_velocity(state.W_v[i_neigh][k], A_f, alpha_j, state.fluid->rho_g());
//             double Re = __Reynolds(G_m_i, D_H_i, state.fluid->mu(h_l_i));
//             double X_bar = __quality_avg(G_m_i, G_m_j);
//             double tp_mult = __two_phase_multiplier(X_bar, Re);
//             double eddy_V = __eddy_velocity(Re, S_ij, D_H_i, D_H_j, D_rod, G_m_i, X, state.fluid->rho_m(X));

//             // turbulent mixing liquid mass transfer
//             state.G_l_tm[i][k_node][ns] = eddy_V * tp_mult * (
//                 (1 - alpha_i) * rho_l_i - (1 - alpha_j) * rho_l_j
//             ); // Eq. 37 from ANTS Theory

//             // turbulent mixing vapor mass transfer
//             state.G_v_tm[i][k_node][ns] = eddy_V * tp_mult * state.fluid->rho_g() * (alpha_i - alpha_j); // Eq. 38 from ANTS Theory

//             // turbulent mixing energy transfer
//             state.Q_m_tm[i][k_node][ns] = eddy_V * tp_mult * (
//                     (1 - alpha_i) * rho_l_i * h_l_i + alpha_i * state.fluid->rho_g() * state.fluid->h_g()
//                 - (1 - alpha_j) * rho_l_j * h_l_j - alpha_j * state.fluid->rho_g() * state.fluid->h_g()
//             ); // Eq. 39 from ANTS Theory

//             // turbulent mixing momentum transfer
//             state.M_m_tm[i][k_node][ns] = eddy_V * tp_mult * (G_m_i - G_m_j); // Eq. 40 from ANTS Theory
//         }
//     }
// }

// void TH::solve_void_drift(State& state) {

//     size_t k = state.surface_plane;  // closure relations use lagging edge values
//     size_t k_node = state.node_plane;

//     Vector2D rho_l = state.fluid->rho(state.h_l);

//     double A_f = state.geom->flow_area(); // flow area [m^2]
//     double D_rod = state.geom->heated_perimeter() / M_PI; // rod diameter [m], assuming square array
//     double D_H_i = state.geom->hydraulic_diameter(); // hydraulic diameter of subchannel i [m]
//     double D_H_j = state.geom->hydraulic_diameter(); // hydraulic diameter of subchannel j [m]
//     double S_ij = state.geom->gap_width(); // gap width between subchannels [m]

//     size_t i_neigh;
//     size_t j_neigh;

//     // loop over subchannel nodes
//     for (size_t i = 0; i < state.geom->nchannels(); ++i) {
//         // loop over 4 surfaces between neighboring subchannels; 0: west, 1: east, 2: north, 3: south
//         for (size_t ns = 0; ns < 4; ++ns) {
//             // perform calculations for surface ns between subchannels i and j
//             size_t i_neigh; size_t j_neigh;
//             std::tie(i_neigh, j_neigh) = __get_neighbor_ij(i, j, ns, state.geom->nx(), state.geom->ny());

//             if ((i_neigh == -1) || (j_neigh == -1)) {
//                 continue; // skip if neighbor is out of bounds
//             }

//             double X = state.X[i][k]; // quality in subchannel i
//             double G_m_i = state.W_m(i, k) / A_f;
//             double G_m_j = state.W_m(i_neigh, k) / A_f;
//             double rho_l_i = rho_l[i][k];
//             double rho_l_j = rho_l[i_neigh][k];
//             double h_l_i = state.h_l[i][k];
//             double h_l_j = state.h_l[i_neigh][k];
//             double alpha_i = state.alpha[i][k];
//             double alpha_j = state.alpha[i_neigh][k];
//             double V_l_i = __liquid_velocity(state.W_l[i][k], A_f, alpha_i, rho_l_i);
//             double V_l_j = __liquid_velocity(state.W_l[i_neigh][k], A_f, alpha_j, rho_l_j);
//             double V_v_i = __vapor_velocity(state.W_v[i][k], A_f, alpha_i, state.fluid->rho_g());
//             double V_v_j = __vapor_velocity(state.W_v[i_neigh][k], A_f, alpha_j, state.fluid->rho_g());
//             double Re = __Reynolds(G_m_i, D_H_i, state.fluid->mu(h_l_i));
//             double X_bar = __quality_avg(G_m_i, G_m_j);
//             double tp_mult = __two_phase_multiplier(X_bar, Re);
//             double eddy_V = __eddy_velocity(Re, S_ij, D_H_i, D_H_j, D_rod, G_m_i, X, state.fluid->rho_m(X));

//             // void drift liquid mass transfer
//             state.G_l_vd[i][k_node][ns] = eddy_V * tp_mult * X_bar * (
//                 alpha_i * rho_l_i + alpha_j * rho_l_j
//             ); // Eq. 42 from ANTS Theory

//             // void drift vapor mass transfer
//             state.G_v_vd[i][k_node][ns] = -eddy_V * tp_mult * X_bar * (
//                 alpha_i + alpha_j
//             ) * state.fluid->rho_g(); // Eq. 41 from ANTS Theory

//             // void drift energy transfer
//             state.Q_m_vd[i][k_node][ns] = eddy_V * tp_mult * X_bar * (
//                 alpha_i * rho_l_i * h_l_i + alpha_j * rho_l_j * h_l_j
//                 - (alpha_i + alpha_j) * state.fluid->rho_g() * state.fluid->h_g()
//             ); // Eq. 43 from ANTS Theory

//             // void drift momentum transfer
//             state.M_m_vd[i][k_node][ns] = eddy_V * tp_mult * X_bar * (
//                 alpha_i * rho_l_i * V_l_i + alpha_j * rho_l_j * V_l_j
//                 - (alpha_i * V_v_i + alpha_j * V_v_j) * state.fluid->rho_g()
//             ); // Eq. 44 from ANTS Theory
//         }
//     }
// }

void TH::solve_surface_mass_flux(State& state) {

    const size_t nchan = state.geom->nx() * state.geom->ny();
    const size_t nsurf = state.geom->nsurfaces();
    const size_t k = state.node_plane;
    const double gtol = 1e-6; // mass flux perturbation amount
    const double dz = state.geom->dz();
    const double S_ij = state.geom->gap_width();
    const double aspect = state.geom->aspect_ratio();

    // outer loop for newton iteration convergence

    // mixture outflow, vapor outflow, and mixture energy outflow
    Vector1D wbarms(nchan);
    Vector1D wbarvs(nchan);
    Vector1D whbars(nchan);

    // vectors for solution variables
    Vector2D gk(nsurf, Vector1D(state.geom->naxial())); // total transverse mass flux
    Vector2D hm(nchan, Vector1D(state.geom->naxial())); // mixture enthalpy
    Vector1D glmix(nsurf);   // liq. turbulent mixing
    Vector1D gvmix(nsurf);   // vap. turbulent mixing
    Vector1D gldrift(nsurf); // liq. void drift
    Vector1D gvdrift(nsurf); // vap. void drift
    Vector1D hmix(nsurf);    // turbulent mixing energy transfer
    Vector1D hdrift(nsurf);  // void drift energy transfer

    // Vectors for temporary reference solution variables
    Vector1D gk0(nsurf);

    // Residual vectors and Jacobian Matrix
    Vector1D f0(nsurf);
    Vector1D f3(nsurf);
    Vector2D dfdg(nsurf, Vector1D(nsurf));

    // calculate wbarms, wbarvs, whbars for isurf, jsurf with donored properties
    for (auto& surf : state.geom->surfaces) {
        size_t ns = surf.idx;
        size_t idonor;
        if (surf.G > 0) idonor = surf.from_node;
        else idonor = surf.to_node;
        wbarms[surf.from_node] += dz * S_ij * (gk[ns][k] + glmix[ns] + gvmix[ns] + gldrift[ns] + gvdrift[ns]);
        wbarms[surf.to_node]   -= dz * S_ij * (gk[ns][k] + glmix[ns] + gvmix[ns] + gldrift[ns] + gvdrift[ns]);
        wbarvs[surf.from_node] += dz * S_ij * (gk[ns][k] + gvmix[ns] + gvdrift[ns]);
        wbarvs[surf.to_node]   -= dz * S_ij * (gk[ns][k] + gvmix[ns] + gvdrift[ns]);
        whbars[surf.from_node] += dz * S_ij * (gk[ns][k] * hm[idonor][k-1] + hmix[ns] + hdrift[ns]);
        whbars[surf.to_node]   -= dz * S_ij * (gk[ns][k] * hm[idonor][k-1] + hmix[ns] + hdrift[ns]);
    }

    // some sort of boundary conditions loop ?

    // PLANAR solve

    // calculate the residual vector f0
    for (size_t ns = 0; ns < nsurf; ++ns) {
        f0[ns] = 0.0;
    }

    // store reference values prior to perturbations
    for (size_t ns = 0; ns < nsurf; ++ns) {
        gk0[ns] = gk[ns][k];
    }

    // check convergence and exit early if criteria is met

    std::cout << "\nJacobian Matrix: at plane: " << state.surface_plane << std::endl;
    for (size_t ns1 = 0; ns1 < nsurf; ++ns1) {

        // reset reference values for all variables
        for (size_t ns0 = 0; ns0 < nsurf; ++ns0) {
            gk[ns0][k] = gk0[ns0];
        }

        // perturb the mass flux at surface ns1
        if (gk[ns1][k] > 0) gk[ns1][k] -= gtol;
        else gk[ns1][k] += gtol;

        // PLANAR_PERTURB solve

        for (size_t ns = 0; ns < nsurf; ++ns) {
            size_t idonor;
            if (gk[ns][k] > 0) idonor = state.geom->surfaces[ns].from_node;
            else idonor = state.geom->surfaces[ns].to_node;

            f3[ns] = 0.0;
            dfdg[ns][ns1] = (f3[ns] - f0[ns]) / (gk[ns1][k] - gk0[ns1]);
            std::cout << std::setw(8) << dfdg[ns][ns1];
        }
        std::cout << std::endl;
    }

    // solve the system of equations (overwrites f0 as solution vector)
    solve_linear_system(nsurf, dfdg, f0);

    std::cout << "Linear system solution: dG" << std::endl;

    // update mass fluxes from solution
    for (size_t ns = 0; ns < nsurf; ++ns) {
        std::cout << f0[ns] << std::endl;
        gk[ns][k] -= f0[ns];
    }

    // } // end outer iteration loop
}

void TH::solve_flow_rates(State& state) {
    // Perform calculations for each surface axial plane
    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {

        double SS_l = 0.0; // sum of the liquid surface source terms [kg/m^2/s]
        double SS_v = 0.0; // sum of the vapor surface source terms [kg/m^2/s]

        for (size_t ns = 0; ns < 4; ++ns) {
            SS_l += state.geom->gap_width() * (state.G_l_tm[i][k_node][ns] + state.G_l_vd[i][k_node][ns]);
            SS_v += state.geom->gap_width() * (state.G_v_tm[i][k_node][ns] + state.G_v_vd[i][k_node][ns]);
        }

        state.W_l[i][k] = state.W_l[i][k-1] - state.geom->dz() * (state.evap[i][k_node] + SS_l); // Eq. 61 from ANTS Theory
        state.W_v[i][k] = state.W_v[i][k-1] + state.geom->dz() * (state.evap[i][k_node] - SS_v); // Eq. 62 from ANTS Theory
    }
}

void TH::solve_enthalpy(State& state) {
    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {

        double SS_m = 0.0; // sum of the mixture surface source terms [W/m^2]
        for (size_t ns = 0; ns < 4; ++ns) {
            SS_m += state.geom->gap_width() * (state.Q_m_tm[i][k_node][ns] + state.Q_m_vd[i][k_node][ns]);
        }

        state.h_l[i][k] = (
            (state.W_v[i][k-1] - state.W_v[i][k]) * state.fluid->h_g()
            + state.W_l[i][k-1] * state.h_l[i][k-1] + state.geom->dz() * state.lhr[i][k_node]
            - state.geom->dz() * SS_m
        ) / state.W_l[i][k]; // Eq. 63 from ANTS Theory
    }
}

void TH::solve_void_fraction(State& state) {
    const double tol = 1e-6;
    const double eps = 1e-12; // small number to prevent division by zero

    // based on the Chexal-Lellouche drift flux model
    double P = state.P[0][0]; // assuming constant pressure for simplicity
    double A = state.geom->flow_area();
    double D_h = state.geom->hydraulic_diameter();

    size_t k = state.surface_plane;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double Gv = state.W_v[i][k] / A; // vapor mass flux
        double Gl = state.W_l[i][k] / A; // liquid mass flux

        if (Gv < eps) {
            state.alpha[i][k] = 0.0;
            continue;
        }

        double h_l = state.h_l[i][k];
        double h_v = state.fluid->h_f() + state.X[i][k] * state.fluid->h_fg();
        double rho_g = state.fluid->rho_g();
        double rho_f = state.fluid->rho_f();
        double rho_l = state.fluid->rho(h_l);
        double mu_v = state.fluid->mu(h_v);
        double mu_l = state.fluid->mu(h_l);
        double sigma = state.fluid->sigma();

        double Re_g = state.W_v[i][k] * D_h / (A * mu_v); // local vapor Reynolds number
        double Re_f = state.W_l[i][k] * D_h / (A * mu_l); // local liquid Reynolds number
        double Re;
        if (Re_g > Re_f) {
            Re = Re_g;
        } else {
            Re = Re_f;
        }
        double A1 = 1 / (1 + exp(-Re / 60000));
        double B1 = std::min(0.8, A1); // from Zuber correlation
        double B2 = 1.41;

        auto f = [A1, B1, B2, P, rho_g, rho_f, rho_l, sigma, Gv, Gl] (double alpha) {

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
            if (fa * fb > 0) throw std::runtime_error("Root not bracketed!");

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

        state.alpha[i][k] = bisection(f, 0.0, 1.0, tol, 100); // solve for void fraction using bisection method
    }
}

void TH::solve_quality(State& state) {
    size_t k = state.surface_plane;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double G_v = state.W_v[i][k] / state.geom->flow_area(); // vapor mass flux, Eq. 8 from ANTS Theory
        double G_l = state.W_l[i][k] / state.geom->flow_area(); // liquid mass flux, Eq. 9 from ANTS Theory
        state.X[i][k] = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory
    }
}

void TH::solve_pressure(State& state) {

    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    double dz = state.geom->dz();
    double D_h = state.geom->hydraulic_diameter();
    double A_f = state.geom->flow_area();

    Vector2D rho = state.fluid->rho(state.h_l);
    Vector2D mu = state.fluid->mu(state.h_l);

    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {

        // ----- two-phase acceleration pressure drop -----
        double dP_accel = (state.W_m(i, k) * state.V_m(i, k) - state.W_m(i, k-1) * state.V_m(i, k-1)) / A_f;

        // ----- two-phase frictional pressure drop -----
        // mass flux
        double G = state.W_l[i][k] / A_f;
        // Reynolds number
        double Re = __Reynolds(G, D_h, mu[i][k]);

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
        double phi2_ch = 1.0 + (gamma * gamma - 1.0) * (b * pow(state.X[i][k], 0.9) * pow((1.0 - state.X[i][k]), 0.9) + pow(state.X[i][k], 1.8));

        // two-phase wall shear pressure drop, Eq. 29 from ANTS Theory
        double dP_wall_shear = K * G * G / (2.0 * rho[i][k]) * phi2_ch;

        // form loss coefficient (no form losses in this simple model)
        double K_loss = 0.0;

        // two-phase multiplier for form losses (homogeneous), Eq. 35 from ANTS Theory
        double phi2_hom = 1.0 + state.X[i][k] * (state.fluid->rho_f() / state.fluid->rho_g() - 1.0);

        // two-phase geometry form loss pressure drop, Eq. 36 from ANTS Theory
        double dP_form = K_loss * G * G / (2.0 * rho[i][k]) * phi2_hom;

        // two-phase frictional pressure drop, Eq. 36 from ANTS Theory
        double dP_tpfric = dP_wall_shear + dP_form;

        // ----- two-phase gravitational pressure drop -----
        double dP_grav = rho[i][k] * g * dz;

        // ----- momentum exchange due to pressure-directed crossflow, turbulent mixing, and void drift -----
        double TM_SS = 0.0; // sum of turbulent mixing liquid momentum exchange terms [Pa]
        double VD_SS = 0.0; // sum of void drift liquid momentum exchange terms [Pa]
        for (size_t ns = 0; ns < 4; ++ns) {
            TM_SS += state.geom->gap_width() * state.M_m_tm[i][k_node][ns];
            VD_SS += state.geom->gap_width() * state.M_m_vd[i][k_node][ns];
        }
        double dP_CF = 0.0; // implement to complete Issue #73
        double dP_TM = dz / A_f * TM_SS;
        double dP_VD = dz / A_f * VD_SS;
        double dP_momexch = dP_CF + dP_TM + dP_VD;

        // ----- total pressure drop over this axial plane -----
        double dP_total = dP_accel + dP_tpfric + dP_grav + dP_momexch; // Eq. 65 from ANTS Theory
        state.P[i][k] = state.P[i][k-1] - dP_total;
    }
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
        return W_l / (A_f * (1 - alpha) * rho_l);
    }
    return 0.0;
}

double TH::__vapor_velocity(double W_v, double A_f, double alpha, double rho_g) {
    if (alpha > 0.0) {
        return W_v / (A_f * alpha * rho_g);
    }
    return 0.0;
}

double TH::__eddy_velocity(double Re, double S_ij, double D_H_i, double D_H_j, double D_rod, double G_m_i, double X, double rho_m) {
    double lambda = 0.0058 * (S_ij / D_rod); // Eq. 46 from ANTS Theory
    return 0.5 * lambda * pow(Re, -0.1) * (1.0 + pow(D_H_j / D_H_i, 1.5)) * D_H_i / D_rod * G_m_i / rho_m; // Eq. 45 from ANTS Theory
}

double TH::__two_phase_multiplier(double X_bar, double Re) {
    double Theta_M = 5.0; // constant set equal to 5.0 for BWR applications, from ANTS Theory
    double X_0_X_M = 0.57 * pow(Re, 0.0417); // Eq. 48 from ANTS Theory
    if (X_bar <= 1.0) {
        return 1.0 + (Theta_M - 1.0) * X_bar; // Eq. 47 from ANTS Theory
    } else {
        return 1.0 + (Theta_M - 1.0) * (1.0 - X_0_X_M) / (X_bar - X_0_X_M); // Eq. 47 from ANTS Theory
    }
}

double TH::__quality_avg(double G_m_i, double G_m_j) {
    double K_M = 1.4; // constant from ANTS Theory, referenced from Lahey and Moody (1977)
    return K_M * (G_m_i - G_m_j) / (G_m_i + G_m_j); // Eq. 49 from ANTS Theory
}
