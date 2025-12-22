#pragma once

#include <Kokkos_Core.hpp>

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct State {
    using View1D = Kokkos::View<double*, ExecutionSpace>;
    using View2D = Kokkos::View<double**, ExecutionSpace>;

    size_t surface_plane = 0;       // current surface axial plane being solved
    size_t node_plane = 0;          // current node axial plane being solved
    size_t max_outer_iter;          // maximum outer iterations
    size_t max_inner_iter;          // maximum inner iterations
    std::shared_ptr<Water<ExecutionSpace>> fluid;   // reference to fluid properties
    std::shared_ptr<Geometry<ExecutionSpace>> geom; // reference to geometry

    View2D h_l;       // liquid enthalpy
    View2D W_l;       // liquid mass flow rate
    View2D W_v;       // vapor mass flow rate
    View2D P;         // pressure
    View2D alpha;     // void fraction
    View2D X;         // quality
    View2D lhr;       // linear heat rate
    View2D evap;      // evaporation term [kg/m/s]

    View1D G_l_tm;    // turbulent mixing liquid mass transfer [kg/m^2/s]
    View1D G_v_tm;    // turbulent mixing vapor mass transfer [kg/m^2/s]
    View1D Q_m_tm;    // turbulent mixing energy transfer [W/m^2]
    View1D M_m_tm;    // turbulent mixing momentum transfer [Pa]

    View1D G_l_vd;    // void drift liquid mass transfer [kg/m^2/s]
    View1D G_v_vd;    // void drift vapor mass transfer [kg/m^2/s]
    View1D Q_m_vd;    // void drift energy transfer [W/m^2]
    View1D M_m_vd;    // void drift momentum transfer [Pa]

    View2D gk;        // surface mass fluxes [kg/m/s]

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
          h_l("h_l", other.h_l.extent(0), other.h_l.extent(1)),
          W_l("W_l", other.W_l.extent(0), other.W_l.extent(1)),
          W_v("W_v", other.W_v.extent(0), other.W_v.extent(1)),
          P("P", other.P.extent(0), other.P.extent(1)),
          alpha("alpha", other.alpha.extent(0), other.alpha.extent(1)),
          X("X", other.X.extent(0), other.X.extent(1)),
          lhr("lhr", other.lhr.extent(0), other.lhr.extent(1)),
          evap("evap", other.evap.extent(0), other.evap.extent(1)),
          G_l_tm("G_l_tm", other.G_l_tm.extent(0)),
          G_v_tm("G_v_tm", other.G_v_tm.extent(0)),
          Q_m_tm("Q_m_tm", other.Q_m_tm.extent(0)),
          M_m_tm("M_m_tm", other.M_m_tm.extent(0)),
          G_l_vd("G_l_vd", other.G_l_vd.extent(0)),
          G_v_vd("G_v_vd", other.G_v_vd.extent(0)),
          Q_m_vd("Q_m_vd", other.Q_m_vd.extent(0)),
          M_m_vd("M_m_vd", other.M_m_vd.extent(0)),
          gk("gk", other.gk.extent(0), other.gk.extent(1)) {
        Kokkos::deep_copy(h_l, other.h_l);
        Kokkos::deep_copy(W_l, other.W_l);
        Kokkos::deep_copy(W_v, other.W_v);
        Kokkos::deep_copy(P, other.P);
        Kokkos::deep_copy(alpha, other.alpha);
        Kokkos::deep_copy(X, other.X);
        Kokkos::deep_copy(lhr, other.lhr);
        Kokkos::deep_copy(evap, other.evap);
        Kokkos::deep_copy(G_l_tm, other.G_l_tm);
        Kokkos::deep_copy(G_v_tm, other.G_v_tm);
        Kokkos::deep_copy(Q_m_tm, other.Q_m_tm);
        Kokkos::deep_copy(M_m_tm, other.M_m_tm);
        Kokkos::deep_copy(G_l_vd, other.G_l_vd);
        Kokkos::deep_copy(G_v_vd, other.G_v_vd);
        Kokkos::deep_copy(Q_m_vd, other.Q_m_vd);
        Kokkos::deep_copy(M_m_vd, other.M_m_vd);
        Kokkos::deep_copy(gk, other.gk);
    }

    // copy assignment
    State& operator=(const State& other) {
        if (this != &other) {
            surface_plane = other.surface_plane;
            node_plane = other.node_plane;
            max_outer_iter = other.max_outer_iter;
            max_inner_iter = other.max_inner_iter;
            fluid = other.fluid;
            geom = other.geom;

            // Resize and copy views
            Kokkos::resize(h_l, other.h_l.extent(0), other.h_l.extent(1));
            Kokkos::resize(W_l, other.W_l.extent(0), other.W_l.extent(1));
            Kokkos::resize(W_v, other.W_v.extent(0), other.W_v.extent(1));
            Kokkos::resize(P, other.P.extent(0), other.P.extent(1));
            Kokkos::resize(alpha, other.alpha.extent(0), other.alpha.extent(1));
            Kokkos::resize(X, other.X.extent(0), other.X.extent(1));
            Kokkos::resize(lhr, other.lhr.extent(0), other.lhr.extent(1));
            Kokkos::resize(evap, other.evap.extent(0), other.evap.extent(1));
            Kokkos::resize(G_l_tm, other.G_l_tm.extent(0));
            Kokkos::resize(G_v_tm, other.G_v_tm.extent(0));
            Kokkos::resize(Q_m_tm, other.Q_m_tm.extent(0));
            Kokkos::resize(M_m_tm, other.M_m_tm.extent(0));
            Kokkos::resize(G_l_vd, other.G_l_vd.extent(0));
            Kokkos::resize(G_v_vd, other.G_v_vd.extent(0));
            Kokkos::resize(Q_m_vd, other.Q_m_vd.extent(0));
            Kokkos::resize(M_m_vd, other.M_m_vd.extent(0));
            Kokkos::resize(gk, other.gk.extent(0), other.gk.extent(1));

            Kokkos::deep_copy(h_l, other.h_l);
            Kokkos::deep_copy(W_l, other.W_l);
            Kokkos::deep_copy(W_v, other.W_v);
            Kokkos::deep_copy(P, other.P);
            Kokkos::deep_copy(alpha, other.alpha);
            Kokkos::deep_copy(X, other.X);
            Kokkos::deep_copy(lhr, other.lhr);
            Kokkos::deep_copy(evap, other.evap);
            Kokkos::deep_copy(G_l_tm, other.G_l_tm);
            Kokkos::deep_copy(G_v_tm, other.G_v_tm);
            Kokkos::deep_copy(Q_m_tm, other.Q_m_tm);
            Kokkos::deep_copy(M_m_tm, other.M_m_tm);
            Kokkos::deep_copy(G_l_vd, other.G_l_vd);
            Kokkos::deep_copy(G_v_vd, other.G_v_vd);
            Kokkos::deep_copy(Q_m_vd, other.Q_m_vd);
            Kokkos::deep_copy(M_m_vd, other.M_m_vd);
            Kokkos::deep_copy(gk, other.gk);
        }
        return *this;
    }

    // move constructor and assignment for efficiency
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;

    // mixture enthalpy
    View2D h_m() const {
        if (W_l.extent(0) != W_v.extent(0) || h_l.extent(0) != W_l.extent(0)) {
            throw std::length_error("State::h_m(): h_l, W_l, and W_v have different sizes");
        }
        View2D out("h_m", h_l.extent(0), h_l.extent(1));
        auto h_l_copy = h_l;
        auto X_copy = X;
        auto fluid_ptr = fluid;
        Kokkos::parallel_for("h_m", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h_l.extent(0), h_l.extent(1)}),
            KOKKOS_LAMBDA(const size_t ij, const size_t k) {
                out(ij, k) = (1 - X_copy(ij, k)) * h_l_copy(ij, k) + X_copy(ij, k) * fluid_ptr->h_g();
            });
        return out;
    }

    double h_m(size_t ij, size_t k) const {
        return (1 - X(ij, k)) * h_l(ij, k) + X(ij, k) * fluid->h_g();
    }

    // mixture mass flow rate
    View2D W_m() const {
        if (W_l.extent(0) != W_v.extent(0)) {
            throw std::length_error("State::W_m(): W_l and W_v have different sizes");
        }
        View2D out("W_m", W_l.extent(0), W_l.extent(1));
        auto W_l_copy = W_l;
        auto W_v_copy = W_v;
        Kokkos::parallel_for("W_m", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {W_l.extent(0), W_l.extent(1)}),
            KOKKOS_LAMBDA(const size_t ij, const size_t k) {
                out(ij, k) = W_l_copy(ij, k) + W_v_copy(ij, k);
            });
        return out;
    }

    double W_m(size_t ij, size_t k) const {
        return W_l(ij, k) + W_v(ij, k);
    }

    // mixture velocity
    View2D V_m() const {
        View2D out("V_m", W_l.extent(0), W_l.extent(1));
        for (size_t ij = 0; ij < out.extent(0); ++ij) {
            for (size_t k = 0; k < out.extent(1); ++k) {
                out(ij, k) = V_m(ij, k);
            }
        }
        return out;
    }

    double V_m(size_t ij, size_t k) const {
        View2D rho = fluid->rho(h_l);
        double A_f = geom->flow_area();
        double v_m;
        if (alpha(ij, k) < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha(ij, k) > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X(ij, k)) * (1.0 - X(ij, k)) / ((1.0 - alpha(ij, k)) * rho(ij, k)) + X(ij, k) * X(ij, k) / (alpha(ij, k) * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m * W_m(ij, k) / A_f; // mixture velocity, Eq. 15 from ANTS Theory
    }

    // mixture specific volume
    View2D nu_m() const {
        View2D out("nu_m", W_l.extent(0), W_l.extent(1));
        for (size_t ij = 0; ij < out.extent(0); ++ij) {
            for (size_t k = 0; k < out.extent(1); ++k) {
                out(ij, k) = nu_m(ij, k);
            }
        }
        return out;
    }

    double nu_m(size_t ij, size_t k) const {
        View2D rho = fluid->rho(h_l);
        double v_m;
        if (alpha(ij, k) < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha(ij, k) > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X(ij, k)) * (1.0 - X(ij, k)) / ((1.0 - alpha(ij, k)) * rho(ij, k)) + X(ij, k) * X(ij, k) / (alpha(ij, k) * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m; // mixture specific volume
    }
};
