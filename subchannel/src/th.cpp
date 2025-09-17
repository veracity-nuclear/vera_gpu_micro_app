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

void TH::solve_evaporation_term(State& state, const Geometry& geom, const Water& fluid) {
    double D_h = geom.hydraulic_diameter(); // hydraulic diameter [m]
    double P_H = geom.heated_perimeter(); // heated perimeter [m]
    double A_f = geom.flow_area(); // flow area [m^2]

    Vector1D mu = fluid.mu(state.h_l); // dynamic viscosity [Pa-s]
    Vector1D cond = fluid.k(state.h_l); // thermal conductivity [W/m-K]
    Vector1D Cp = fluid.Cp(state.h_l); // specific heat [J/kg-K]
    Vector1D T = fluid.T(state.h_l); // temperature [K]

    // Perform calculations for each nodal axial plane
    for (size_t k = 0; k < geom.naxial(); ++k) {
        double Re = state.W_l[k] * D_h / (A_f * mu[k]); // Reynolds number
        double Pr = Cp[k] * mu[k] / cond[k]; // Prandtl number
        double Pe = Re * Pr; // Peclet number
        double Qflux_wall = state.lhr[k] / P_H; // wall heat flux [W/m^2]
        double G_m = state.W_m(k) / A_f; // mixture mass flux [kg/m^2-s]

        double void_dc; // void departure, Eq. 52 from ANTS Theory
        if (Pe < 70000.) {
            void_dc = 0.0022 * Pe * (Qflux_wall / G_m);
        } else {
            void_dc = 154.0 * (Qflux_wall / G_m);
        }

        double Qflux_boil; // boiling heat flux [W/m], Eq. 51 from ANTS Theory
        if ((fluid.h_f() - state.h_l[k]) < void_dc) {
            Qflux_boil = Qflux_wall * (1 - ((fluid.h_f() - state.h_l[k]) / void_dc));
        } else {
            Qflux_boil = 0.0;
        }

        double epsilon = fluid.rho_f() * (fluid.h_f() - state.h_l[k]) / (fluid.rho_g() * fluid.h_fg()); // pumping parameter, Eq. 53 from ANTS Theory
        double H_0 = 0.075; // [s^-1 K^-1], condensation parameter; value recommended by Lahey and Moody (1996)
        double gamma_cond = (H_0 * (1 / fluid.v_fg()) * A_f * state.alpha[k] * (fluid.Tsat() - T[k])) / P_H; // condensation rate [kg/m^3-s], Eq. 53 from ANTS Theory

        state.evap[k] = P_H * Qflux_boil / (fluid.h_fg() * (1 + epsilon)) - P_H * gamma_cond; // Eq. 50 from ANTS Theory
    }
}

void TH::solve_turbulent_mixing(State& state, const Geometry& geom, const Water& fluid) {}

void TH::solve_void_drift(State& state, const Geometry& geom, const Water& fluid) {}

void TH::solve_surface_mass_flux(State& state, const Geometry& geom, const Water& fluid) {}

void TH::solve_flow_rates(State& state, const Geometry& geom, const Water& fluid) {
    // Perform calculations for each surface axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        state.W_l[k] = state.W_l[k-1] - geom.dz() * state.evap[k-1]; // Eq. 61 from ANTS Theory
        state.W_v[k] = state.W_v[k-1] + geom.dz() * state.evap[k-1]; // Eq. 62 from ANTS Theory
    }
}

void TH::solve_enthalpy(State& state, const Geometry& geom, const Water& fluid) {
    // Perform calculations for each surface axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        state.h_l[k] = (
            (state.W_v[k-1] - state.W_v[k]) * fluid.h_g() +
            state.W_l[k-1] * state.h_l[k-1] + geom.dz() * state.lhr[k-1]
        ) / state.W_l[k]; // Eq. 63 from ANTS Theory
    }
}

void TH::solve_void_fraction(State& state, const Geometry& geom, const Water& fluid) {
    const size_t maxIter = 1000;
    const double tol = 1e-6;
    const double eps = 1e-12; // small number to prevent division by zero
    Vector1D alpha_prev(state.alpha); // previous iteration void fraction

    // based on the Chexal-Lellouche drift flux model
    double P = state.P[0]; // assuming constant pressure for simplicity
    double A = geom.flow_area();

    for (size_t iter = 0; iter < maxIter; ++iter) { // iterate to converge void fraction
        for (size_t k = 0; k < geom.naxial() + 1; ++k) {
            double G_v = state.W_v[k] / A; // vapor mass flux
            double G_l = state.W_l[k] / A; // liquid mass flux

            // calculate distribution parameter, C_0
            double B_1 = 1.5; // from Zuber correlation
            double B_2 = 1.41;

            // Safeguard against pressure approaching critical pressure
            double denom_C1 = P * (P_crit - P);
            if (std::abs(denom_C1) < eps) {
                denom_C1 = eps; // prevent division by zero
            }
            double C_1 = 4.0 * P_crit * P_crit / denom_C1; // Eq. 24 from ANTS Theory

            // calculate Chexal-Lellouche fluid parameter, L
            double exp_term1 = std::exp(-C_1 * state.alpha[k]);
            double exp_term2 = std::exp(-C_1);
            double L_denom = 1.0 - exp_term2;
            if (std::abs(L_denom) < eps) {
                L_denom = eps; // prevent division by zero
            }
            double L = (1.0 - exp_term1) / L_denom; // Eq. 23 from ANTS Theory

            double K_0 = B_1 + (1 - B_1) * pow(fluid.rho_g() / fluid.rho_f(), 0.25); // Eq. 25 from ANTS Theory
            double r = (1 + 1.57 * (fluid.rho_g() / fluid.rho_f())) / (1 - B_1); // Eq. 26 from ANTS Theory

            // Safeguard against invalid power operations
            double alpha_safe = std::max(0.0, std::min(state.alpha[k], 0.99));
            double C_0_denom = K_0 + (1 - K_0) * pow(alpha_safe, r);
            if (std::abs(C_0_denom) < eps) {
                C_0_denom = eps; // prevent division by zero
            }
            double C_0 = L / C_0_denom; // Eq. 22 from ANTS Theory

            // calculate drift velocity, V_gj
            double V_gj0 = B_2 * pow(((fluid.rho_f() - fluid.rho_g()) * g * fluid.sigma()) / (fluid.mu_f() * fluid.mu_f()), 0.25); // Eq. 28 from ANTS Theory
            double alpha_drift = std::max(0.0, std::min(1.0 - state.alpha[k], 1.0));
            double V_gj = V_gj0 * pow(alpha_drift, B_1); // Eq. 27 from ANTS Theory

            // update void fraction with safeguards
            double numerator = G_v;
            double denominator = C_0 * (G_v + (fluid.rho_g() / fluid.rho_f()) * G_l) + fluid.rho_g() * V_gj;
            if (std::abs(denominator) < eps) {
                denominator = eps; // prevent division by zero
            }
            state.alpha[k] = numerator / denominator; // Eq. 21 from ANTS Theory

        }

        double max_diff = 0.0; // calculate max change in alpha for convergence
        for (size_t k = 0; k < geom.naxial() + 1; ++k) {
            double diff = std::abs(state.alpha[k] - alpha_prev[k]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
        if (max_diff < tol) {
            break;
        }
        alpha_prev = state.alpha; // update previous alpha for next iteration
    }
}

void TH::solve_quality(State& state, const Geometry& geom, const Water& fluid) {
    for (size_t k = 0; k < geom.naxial() + 1; ++k) {
        double G_v = state.W_v[k] / geom.flow_area(); // vapor mass flux, Eq. 8 from ANTS Theory
        double G_l = state.W_l[k] / geom.flow_area(); // liquid mass flux, Eq. 9 from ANTS Theory
        state.X[k] = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory
    }
}

void TH::solve_pressure(State& state, const Geometry& geom, const Water& fluid) {

    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    Vector1D rho = fluid.rho(state.h_l);
    Vector1D mu = fluid.mu(state.h_l);

    // start at k=1 because inlet pressure is a constant boundary condition
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        // calculate mass flux
        double G = state.W_l[k] / geom.flow_area();

        // calculate Reynolds number
        double Re = state.W_l[k] * geom.hydraulic_diameter() / (geom.flow_area() * mu[k]);

        // calculate frictional pressure drop from wall shear
        double f = a_1 * pow(Re, n); // single phase friction factor, Eq. 31 from ANTS Theory
        double K = f * geom.dz() / geom.hydraulic_diameter(); // frictional loss coefficient, Eq. 30 from ANTS Theory
        double gamma = pow(fluid.rho_f() / fluid.rho_g(), 0.5) * pow(fluid.mu_g() / fluid.mu_f(), 0.2); // Eq. 33 from ANTS Theory

        // calculate parameter b for two-phase multiplier (Chisholm), Eq. 34 from ANTS Theory
        double b;
        if (gamma <= 9.5) {
            b = 55.0 / sqrt(G);
        } else if (gamma < 28) {
            b = 520.0 / (gamma * sqrt(G));
        } else {
            b = 15000.0 / (gamma * gamma * sqrt(G));
        }

        // calculate two-phase multiplier for wall shear (Chisholm), Eq. 32 from ANTS Theory
        double phi2_ch = 1.0 + (gamma * gamma - 1.0) * (b * pow(state.X[k], 0.9) * pow((1.0 - state.X[k]), 0.9) + pow(state.X[k], 1.8));

        // calculate two-phase wall shear pressure drop, Eq. 29 from ANTS Theory
        double dP_wall_shear = K * G * G / (2.0 * rho[k]) * phi2_ch;

        // form loss coefficient (no form losses in this simple model)
        double K_loss = 0.0;

        // calculate two-phase multiplier for form losses (homogeneous), Eq. 35 from ANTS Theory
        double phi2_hom = 1.0 + state.X[k] * (fluid.rho_f() / fluid.rho_g() - 1.0);

        // calculate two-phase geometry form loss pressure drop, Eq. 36 from ANTS Theory
        double dP_form = K_loss * G * G / (2.0 * rho[k]) * phi2_hom;

        // calculate two-phase gravitational pressure drop
        double dP_grav = rho[k] * g * geom.dz();

        // calculate total two-phase pressure drop, Eq. 36 from ANTS Theory
        double dP_total = dP_wall_shear + dP_form + dP_grav;

        // update pressure at this axial plane
        state.P[k] = state.P[k-1] - dP_total;
    }
}
