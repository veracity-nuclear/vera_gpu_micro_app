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

void TH::solve_enthalpy(State& state, const Geometry& geom) {

    // Perform calculations for each axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        state.h[k] = (state.W_l[k] * state.h[k-1] + geom.dz() * state.lhr[k-1]) / state.W_l[k]; // Eq. 63 from ANTS Theory
    }
}

void TH::solve_pressure(State& state, const Geometry& geom, const Water& fluid) {

    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    Vector1D rho = fluid.rho(state.h);
    Vector1D mu = fluid.mu(state.h);

    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        // calculate mass flux
        double G = state.W_l[k] / geom.flow_area();

        // calculate Reynolds number
        double Re = state.W_l[k] * geom.hydraulic_diameter() / (geom.flow_area() * mu[k]);

        // calculate frictional pressure drop from wall shear
        double f = a_1 * pow(Re, n); // single phase friction factor, Eq. 31 from ANTS Theory
        double K = f * geom.dz() / geom.hydraulic_diameter(); // frictional loss coefficient, Eq. 30 from ANTS Theory
        double b;
        double gamma = pow(fluid.rho_f() / fluid.rho_g(), 0.5) * pow(fluid.mu_g() / fluid.mu_f(), 0.2); // Eq. 33 from ANTS Theory

        // calculate parameter b for two-phase multiplier (Chisholm), Eq. 34 from ANTS Theory
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

void TH::solve_void_fraction(State& state, const Geometry& geom, const Water& fluid) {
    const size_t maxIter = 100;
    const double tol = 1e-6;
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
            double C_1 = 4.0 * P_crit * P_crit / (P * (P_crit - P)); // Eq. 24 from ANTS Theory
            double L = (1 - exp(-C_1) * state.alpha[k]) / (1 - exp(-C_1)); // Eq. 23 from ANTS Theory
            double K_0 = B_1 + (1 - B_1) * pow(fluid.rho_g() / fluid.rho_f(), 0.25); // Eq. 25 from ANTS Theory
            double r = (1 + 1.57 * (fluid.rho_g() / fluid.rho_f())) / (1 - B_1); // Eq. 26 from ANTS Theory
            double C_0 = L / (K_0 + (1 - K_0) * pow(state.alpha[k], r)); // Eq. 22 from ANTS Theory

            // calculate drift velocity, V_gj
            double V_gj0 = B_2 * pow(((fluid.rho_f() - fluid.rho_g()) * g * fluid.sigma()) / (fluid.mu_f() * fluid.mu_f()), 0.25); // Eq. 28 from ANTS Theory
            double V_gj = V_gj0 * pow(1 - state.alpha[k], B_1); // Eq. 27 from ANTS Theory

            // update void fraction
            state.alpha[k] = G_v / (C_0 * (G_v + (fluid.rho_g() / fluid.rho_f()) * G_l) + fluid.rho_g() * V_gj); // Eq. 21 from ANTS Theory
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
        double G_v = state.W_v[k] / geom.flow_area(); // vapor mass flux
        double G_f = state.W_l[k] / geom.flow_area(); // liquid mass flux
        state.X[k] = G_v / (G_v + G_f); // Eq. 17 from ANTS Theory
    }
}
