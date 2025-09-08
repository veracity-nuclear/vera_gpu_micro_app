#include "th.hpp"

void TH::solve_enthalpy(State& state, double lhr, const Geometry& geom) {

    // Perform calculations for each axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        state.h[k] = (state.W_l[k] * state.h[k-1] + geom.dz() * lhr) / state.W_l[k];
    }
}

void TH::solve_pressure(State& state, const Geometry& geom, const Water& fluid) {

    double G; // mass flux
    double Re; // Reynolds number
    double f; // single phase friction factor
    double K; // frictional loss coefficient
    double K_loss; // form loss coefficient
    double gamma;
    double b;
    double phi2_ch; // two-phase multiplier for wall shear (Chisholm)
    double phi2_hom; // two-phase multiplier for form losses (homogeneous)
    double dP_wall_shear; // two-phase wall shear pressure drop
    double dP_form; // two-phase geometry form loss pressure drop
    double dP_grav; // two-phase gravitational pressure drop
    double dP_total; // total two-phase pressure drop

    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    // temporarily set quality to zero
    double X_f = 0.0;

    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        G = state.W_l[k] / geom.flow_area();

        Vector1D rho = fluid.rho(state.h);
        Vector1D mu = fluid.mu(state.h);

        Re = state.W_l[k] * geom.hydraulic_diameter() / (geom.flow_area() * mu[k]);

        // calculate frictional pressure drop from wall shear
        f = a_1 * pow(Re, n);
        K = f * geom.dz() / geom.hydraulic_diameter();
        gamma = pow(fluid.rho_f() / fluid.rho_g(), 0.5) * pow(fluid.mu_g() / fluid.mu_f(), 0.2);
        if (gamma <= 9.5) {
            b = 55.0 / sqrt(G);
        } else if (gamma > 9.5 && gamma < 28) {
            b = 520.0 / (gamma * sqrt(G));
        } else {
            b = 15000.0 / (gamma * gamma * sqrt(G));
        }
        phi2_ch = 1.0 + (gamma * gamma - 1.0) * (b * pow(X_f, 0.9) * pow((1.0 - X_f), 0.9) + pow(X_f, 1.8));
        dP_wall_shear = K * G * G / (2.0 * rho[k]) * phi2_ch;

        // calculate form loss pressure drop
        K_loss = 0.0; // no form losses in this simple model
        phi2_hom = 1.0 + X_f * (fluid.rho_f() / fluid.rho_g() - 1.0);
        dP_form = K_loss * G * G / (2.0 * rho[k]) * phi2_hom;

        // calculate gravitational pressure drop
        dP_grav = rho[k] * 9.81 * geom.dz();

        dP_total = dP_wall_shear + dP_form + dP_grav;

        state.P[k] = state.P[k-1] - dP_total;
    }
}
