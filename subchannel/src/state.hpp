#pragma once

#include "vectors.hpp"

struct State {
    size_t surface_plane = 0;       // current surface axial plane being solved
    size_t node_plane = 0;          // current node axial plane being solved
    size_t max_outer_iter;          // maximum outer iterations
    size_t max_inner_iter;          // maximum inner iterations
    std::shared_ptr<Water> fluid;   // reference to fluid properties
    std::shared_ptr<Geometry> geom; // reference to geometry

    Vector2D h_l;       // liquid enthalpy
    Vector2D W_l;       // liquid mass flow rate
    Vector2D W_v;       // vapor mass flow rate
    Vector2D P;         // pressure
    Vector2D alpha;     // void fraction
    Vector2D X;         // quality
    Vector2D lhr;       // linear heat rate
    Vector2D evap;      // evaporation term [kg/m/s]

    Vector1D G_l_tm;    // turbulent mixing liquid mass transfer [kg/m^2/s]
    Vector1D G_v_tm;    // turbulent mixing vapor mass transfer [kg/m^2/s]
    Vector1D Q_m_tm;    // turbulent mixing energy transfer [W/m^2]
    Vector1D M_m_tm;    // turbulent mixing momentum transfer [Pa]

    Vector1D G_l_vd;    // void drift liquid mass transfer [kg/m^2/s]
    Vector1D G_v_vd;    // void drift vapor mass transfer [kg/m^2/s]
    Vector1D Q_m_vd;    // void drift energy transfer [W/m^2]
    Vector1D M_m_vd;    // void drift momentum transfer [Pa]

    Vector2D gk;        // surface mass fluxes [kg/m/s]

    // -------------------------
    // Constructors
    // -------------------------
    State() = default;

    // copy constructor (deep copy of data; shared_ptrs are shared)
    State(const State& other)
        : surface_plane(other.surface_plane),
          node_plane(other.node_plane),
          max_outer_iter(other.max_outer_iter),
          max_inner_iter(other.max_inner_iter),
          fluid(other.fluid), // shared_ptr copy — shares ownership
          geom(other.geom),   // shared_ptr copy — shares ownership
          h_l(other.h_l),
          W_l(other.W_l),
          W_v(other.W_v),
          P(other.P),
          alpha(other.alpha),
          X(other.X),
          lhr(other.lhr),
          evap(other.evap),
          G_l_tm(other.G_l_tm),
          G_v_tm(other.G_v_tm),
          Q_m_tm(other.Q_m_tm),
          M_m_tm(other.M_m_tm),
          G_l_vd(other.G_l_vd),
          G_v_vd(other.G_v_vd),
          Q_m_vd(other.Q_m_vd),
          M_m_vd(other.M_m_vd),
          gk(other.gk) {}

    // copy assignment
    State& operator=(const State& other) {
        if (this != &other) {
            surface_plane = other.surface_plane;
            node_plane = other.node_plane;
            max_outer_iter = other.max_outer_iter;
            max_inner_iter = other.max_inner_iter;
            fluid = other.fluid;
            geom = other.geom;
            h_l = other.h_l;
            W_l = other.W_l;
            W_v = other.W_v;
            P = other.P;
            alpha = other.alpha;
            X = other.X;
            lhr = other.lhr;
            evap = other.evap;
            G_l_tm = other.G_l_tm;
            G_v_tm = other.G_v_tm;
            Q_m_tm = other.Q_m_tm;
            M_m_tm = other.M_m_tm;
            G_l_vd = other.G_l_vd;
            G_v_vd = other.G_v_vd;
            Q_m_vd = other.Q_m_vd;
            M_m_vd = other.M_m_vd;
            gk = other.gk;
        }
        return *this;
    }

    // move constructor and assignment for efficiency
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    // mixture enthalpy
    Vector2D h_m() const {
        if (W_l.size() != W_v.size() || h_l.size() != W_l.size()) {
            throw std::length_error("State::h_m(): h_l, W_l, and W_v have different sizes");
        }
        Vector2D out(h_l.size());
        for (size_t ij = 0; ij < out.size(); ++ij) {
            for (size_t k = 0; k < out[ij].size(); ++k) {
                double X_ijk = X[ij][k];
                out[ij][k] = h_m(ij, k);
            }
        }
        return out;
    }

    double h_m(size_t ij, size_t k) const {
        return (1 - X[ij][k]) * h_l[ij][k] + X[ij][k] * fluid->h_g();
    }

    // mixture mass flow rate
    Vector2D W_m() const {
        if (W_l.size() != W_v.size()) {
            throw std::length_error("State::W_m(): W_l and W_v have different sizes");
        }
        Vector2D out(W_l.size());
        for (size_t ij = 0; ij < out.size(); ++ij) {
            for (size_t k = 0; k < out[ij].size(); ++k) {
                out[ij][k] = W_m(ij, k);
            }
        }
        return out;
    }

    double W_m(size_t ij, size_t k) const {
        return W_l[ij][k] + W_v[ij][k];
    }

    // mixture velocity
    Vector2D V_m() const {
        Vector2D out(W_l.size());
        for (size_t ij = 0; ij < out.size(); ++ij) {
            for (size_t k = 0; k < out[ij].size(); ++k) {
                out[ij][k] = V_m(ij, k);
            }
        }
        return out;
    }

    double V_m(size_t ij, size_t k) const {
        Vector2D rho = fluid->rho(h_l);
        double A_f = geom->flow_area();
        double v_m;
        if (alpha[ij][k] < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha[ij][k] > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X[ij][k]) * (1.0 - X[ij][k]) / ((1.0 - alpha[ij][k]) * rho[ij][k]) + X[ij][k] * X[ij][k] / (alpha[ij][k] * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m * W_m(ij, k) / A_f; // mixture velocity, Eq. 15 from ANTS Theory
    }

    // mixture specific volume
    Vector2D nu_m() const {
        Vector2D out(W_l.size());
        for (size_t ij = 0; ij < out.size(); ++ij) {
            for (size_t k = 0; k < out[ij].size(); ++k) {
                out[ij][k] = nu_m(ij, k);
            }
        }
        return out;
    }

    double nu_m(size_t ij, size_t k) const {
        Vector2D rho = fluid->rho(h_l);
        double v_m;
        if (alpha[ij][k] < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha[ij][k] > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X[ij][k]) * (1.0 - X[ij][k]) / ((1.0 - alpha[ij][k]) * rho[ij][k]) + X[ij][k] * X[ij][k] / (alpha[ij][k] * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m; // mixture specific volume
    }
};
