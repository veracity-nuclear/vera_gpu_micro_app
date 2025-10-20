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

    Vector3D mu = fluid.mu(state.h_l); // dynamic viscosity [Pa-s]
    Vector3D rho = fluid.rho(state.h_l); // liquid density [kg/m^3]
    Vector3D cond = fluid.k(state.h_l); // thermal conductivity [W/m-K]
    Vector3D Cp = fluid.Cp(state.h_l); // specific heat [J/kg-K]
    Vector3D T = fluid.T(state.h_l); // temperature [K]

    // Perform calculations for each nodal axial plane
    for (size_t k = 0; k < geom.naxial(); ++k) {
        for (size_t j = 0; j < geom.ny(); ++j) {
            for (size_t i = 0; i < geom.nx(); ++i) {
                double Re = state.W_l[i][j][k] * D_h / (A_f * mu[i][j][k]); // Reynolds number
                double Pr = Cp[i][j][k] * mu[i][j][k] / cond[i][j][k]; // Prandtl number
                double Pe = Re * Pr; // Peclet number
                double Qflux_wall = state.lhr[i][j][k] / P_H; // wall heat flux [W/m^2]
                double G_l = state.W_l[i][j][k] / A_f; // liquid mass flux [kg/m^2-s]
                double G_v = state.W_v[i][j][k] / A_f; // vapor mass flux [kg/m^2-s]
                double G_m = G_l + G_v; // mixture mass flux [kg/m^2-s], Eq. 11 from ANTS Theory

                double void_dc; // void departure, Eq. 52 from ANTS Theory
                if (Pe < 70000.) {
                    void_dc = 0.0022 * Pe * (Qflux_wall / G_m);
                } else {
                    void_dc = 154.0 * (Qflux_wall / G_m);
                }

                if (state.h_l[i][j][k] < fluid.h_f()) {
                    double Qflux_boil; // boiling heat flux [W/m], Eq. 51 from ANTS Theory
                    if ((fluid.h_f() - state.h_l[i][j][k]) < void_dc) {
                        Qflux_boil = Qflux_wall * (1 - ((fluid.h_f() - state.h_l[i][j][k]) / void_dc));
                    } else {
                        Qflux_boil = 0.0;
                    }

                    double epsilon = rho[i][j][k] * (fluid.h_f() - state.h_l[i][j][k]) / (fluid.rho_g() * fluid.h_fg()); // pumping parameter, Eq. 53 from ANTS Theory
                    double H_0 = 0.075; // [s^-1 K^-1], condensation parameter; value recommended by Lahey and Moody (1996)
                    double gamma_cond = (H_0 * (1 / fluid.v_fg()) * A_f * state.alpha[i][j][k] * (fluid.Tsat() - T[i][j][k])) / P_H; // condensation rate [kg/m^3-s], Eq. 54 from ANTS Theory
                    state.evap[i][j][k] = P_H * Qflux_boil / (fluid.h_fg() * (1 + epsilon)) - P_H * gamma_cond; // Eq. 50 from ANTS Theory

                } else {
                    double Qflux_boil = Qflux_wall;
                    state.evap[i][j][k] = P_H * Qflux_boil / fluid.h_fg();
                }
            }
        }
    }
}

void TH::solve_turbulent_mixing(State& state, const Geometry& geom, const Water& fluid) {}

void TH::solve_void_drift(State& state, const Geometry& geom, const Water& fluid) {}

void TH::solve_surface_mass_flux(State& state, const Geometry& geom, const Water& fluid) {}

void TH::solve_flow_rates(State& state, const Geometry& geom, const Water& fluid) {
    // Perform calculations for each surface axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        for (size_t j = 0; j < geom.ny(); ++j) {
            for (size_t i = 0; i < geom.nx(); ++i) {
                state.W_l[i][j][k] = state.W_l[i][j][k-1] - geom.dz() * state.evap[i][j][k-1]; // Eq. 61 from ANTS Theory
                state.W_v[i][j][k] = state.W_v[i][j][k-1] + geom.dz() * state.evap[i][j][k-1]; // Eq. 62 from ANTS Theory
            }
        }
    }
}

void TH::solve_enthalpy(State& state, const Geometry& geom, const Water& fluid) {
    // Perform calculations for each surface axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        for (size_t j = 0; j < geom.ny(); ++j) {
            for (size_t i = 0; i < geom.nx(); ++i) {
                state.h_l[i][j][k] = (
                    (state.W_v[i][j][k-1] - state.W_v[i][j][k]) * fluid.h_g() +
                    state.W_l[i][j][k-1] * state.h_l[i][j][k-1] + geom.dz() * state.lhr[i][j][k-1]
                ) / state.W_l[i][j][k]; // Eq. 63 from ANTS Theory
            }
        }
    }
}

void TH::solve_void_fraction(State& state, const Geometry& geom, const Water& fluid) {
    const double tol = 1e-6;
    const double eps = 1e-12; // small number to prevent division by zero

    // based on the Chexal-Lellouche drift flux model
    double P = state.P[0][0][0]; // assuming constant pressure for simplicity
    double A = geom.flow_area();
    double D_h = geom.hydraulic_diameter();

    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        for (size_t j = 0; j < geom.ny(); ++j) {
            for (size_t i = 0; i < geom.nx(); ++i) {
                double Gv = state.W_v[i][j][k] / A; // vapor mass flux
                double Gl = state.W_l[i][j][k] / A; // liquid mass flux

                if (Gv < eps) {
                    state.alpha[i][j][k] = 0.0;
                    continue;
                }

                double h_l = state.h_l[i][j][k];
                double h_v = fluid.h_f() + state.X[i][j][k] * fluid.h_fg();
                double rho_g = fluid.rho_g();
                double rho_f = fluid.rho_f();
                double rho_l = fluid.rho(h_l);
                double mu_v = fluid.mu(h_v);
                double mu_l = fluid.mu(h_l);
                double sigma = fluid.sigma();

                double Re_g = state.W_v[i][j][k] * D_h / (A * mu_v); // local vapor Reynolds number
                double Re_f = state.W_l[i][j][k] * D_h / (A * mu_l); // local liquid Reynolds number
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

                state.alpha[i][j][k] = bisection(f, 0.0, 1.0, tol, 100); // solve for void fraction using bisection method
            }
        }
    }
}

void TH::solve_quality(State& state, const Geometry& geom, const Water& fluid) {
    for (size_t k = 0; k < geom.naxial() + 1; ++k) {
        for (size_t j = 0; j < geom.ny(); ++j) {
            for (size_t i = 0; i < geom.nx(); ++i) {
                double G_v = state.W_v[i][j][k] / geom.flow_area(); // vapor mass flux, Eq. 8 from ANTS Theory
                double G_l = state.W_l[i][j][k] / geom.flow_area(); // liquid mass flux, Eq. 9 from ANTS Theory
                state.X[i][j][k] = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory
            }
        }
    }
}

void TH::solve_pressure(State& state, const Geometry& geom, const Water& fluid) {

    // coefficients for Adams correlation from ANTS Theory
    const double a_1 = 0.1892;
    const double n = -0.2;

    double dz = geom.dz();
    double D_h = geom.hydraulic_diameter();
    double A_f = geom.flow_area();

    Vector3D rho = fluid.rho(state.h_l);
    Vector3D mu = fluid.mu(state.h_l);

    // pre-calculate mixture velocities
    Vector3D V_m;
    Vector::resize(V_m, geom.nx(), geom.ny(), geom.naxial() + 1);
    for (size_t k = 0; k < geom.naxial() + 1; ++k) {
        for (size_t j = 0; j < geom.ny(); ++j) {
            for (size_t i = 0; i < geom.nx(); ++i) {
                double X = state.X[i][j][k];
                double alpha = state.alpha[i][j][k];
                double v_m;
                if (alpha < 1e-6) {
                    v_m = 1.0 / fluid.rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
                } else if (alpha > 1.0 - 1e-6) {
                    v_m = 1.0 / fluid.rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
                } else {
                    v_m = (1.0 - X) * (1.0 - X) / ((1.0 - alpha) * fluid.rho_f()) + X * X / (alpha * fluid.rho_g()); // Eq. 16 from ANTS Theory
                }
                V_m[i][j][k] = v_m * state.W_m(i, j, k) / A_f; // mixture velocity, Eq. 15 from ANTS Theory
            }
        }
    }

    // start at k=1 because inlet pressure is a constant boundary condition
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        for (size_t j = 0; j < geom.ny(); ++j) {
            for (size_t i = 0; i < geom.nx(); ++i) {

                // ----- two-phase acceleration pressure drop -----
                double dP_accel = (state.W_m(i, j, k) * V_m[i][j][k] - state.W_m(i, j, k-1) * V_m[i][j][k-1]) / A_f;

                // ----- two-phase frictional pressure drop -----
                // mass flux
                double G = state.W_l[i][j][k] / A_f;
                // Reynolds number
                double Re = state.W_l[i][j][k] * D_h / (A_f * mu[i][j][k]);

                // frictional pressure drop from wall shear
                double f = a_1 * pow(Re, n); // single phase friction factor, Eq. 31 from ANTS Theory
                double K = f * dz / D_h; // frictional loss coefficient, Eq. 30 from ANTS Theory
                double gamma = pow(fluid.rho_f() / fluid.rho_g(), 0.5) * pow(fluid.mu_g() / fluid.mu_f(), 0.2); // Eq. 33 from ANTS Theory

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
                double phi2_ch = 1.0 + (gamma * gamma - 1.0) * (b * pow(state.X[i][j][k], 0.9) * pow((1.0 - state.X[i][j][k]), 0.9) + pow(state.X[i][j][k], 1.8));

                // two-phase wall shear pressure drop, Eq. 29 from ANTS Theory
                double dP_wall_shear = K * G * G / (2.0 * rho[i][j][k]) * phi2_ch;

                // form loss coefficient (no form losses in this simple model)
                double K_loss = 0.0;

                // two-phase multiplier for form losses (homogeneous), Eq. 35 from ANTS Theory
                double phi2_hom = 1.0 + state.X[i][j][k] * (fluid.rho_f() / fluid.rho_g() - 1.0);

                // two-phase geometry form loss pressure drop, Eq. 36 from ANTS Theory
                double dP_form = K_loss * G * G / (2.0 * rho[i][j][k]) * phi2_hom;

                // two-phase frictional pressure drop, Eq. 36 from ANTS Theory
                double dP_tpfric = dP_wall_shear + dP_form;

                // ----- two-phase gravitational pressure drop -----
                double dP_grav = rho[i][j][k] * g * dz;

                // ----- momentum exchange due to pressure-directed crossflow, turbulent mixing, and void drift -----
                double dP_CF = 0.0; // implement to complete Issue #73
                double dP_TM = 0.0; // implement to complete Issue #74
                double dP_VD = 0.0; // implement to complete Issue #75
                double dP_momexch = dP_CF + dP_TM + dP_VD;

                // ----- total pressure drop over this axial plane -----
                double dP_total = dP_accel + dP_tpfric + dP_grav + dP_momexch; // Eq. 65 from ANTS Theory
                state.P[i][j][k] = state.P[i][j][k-1] - dP_total;
            }
        }
    }
}
