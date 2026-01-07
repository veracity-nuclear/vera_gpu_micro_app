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

    Water fluid;
    std::shared_ptr<Geometry<ExecutionSpace>> geom; // reference to geometry

    View2D h_l;         // liquid enthalpy
    View2D W_l;         // liquid mass flow rate
    View2D W_v;         // vapor mass flow rate
    View2D P;           // pressure
    View2D alpha;       // void fraction
    View2D X;           // quality
    View2D lhr;         // linear heat rate
    View2D evap;        // evaporation term [kg/m/s]
    View2D gk;          // surface mass fluxes [kg/m/s]

    View1D G_l_tm;      // turbulent mixing liquid mass transfer [kg/m^2/s]
    View1D G_v_tm;      // turbulent mixing vapor mass transfer [kg/m^2/s]
    View1D Q_m_tm;      // turbulent mixing energy transfer [W/m^2]
    View1D M_m_tm;      // turbulent mixing momentum transfer [Pa]

    View1D G_l_vd;      // void drift liquid mass transfer [kg/m^2/s]
    View1D G_v_vd;      // void drift vapor mass transfer [kg/m^2/s]
    View1D Q_m_vd;      // void drift energy transfer [W/m^2]
    View1D M_m_vd;      // void drift momentum transfer [Pa]

    View1D SS_l;        // accumulated surface source terms for liquid
    View1D SS_v;        // accumulated surface source terms for vapor
    View1D SS_m;        // accumulated surface source terms for mixture enthalpy
    View1D CF_SS;       // crossflow momentum source terms
    View1D TM_SS;       // turbulent mixing momentum source terms
    View1D VD_SS;       // void drift momentum source terms


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
          fluid(other.fluid),
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
          SS_l("SS_l", other.SS_l.extent(0)),
          SS_v("SS_v", other.SS_v.extent(0)),
          SS_m("SS_m", other.SS_m.extent(0)),
          CF_SS("CF_SS", other.CF_SS.extent(0)),
          TM_SS("TM_SS", other.TM_SS.extent(0)),
          VD_SS("VD_SS", other.VD_SS.extent(0)),
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
        Kokkos::deep_copy(SS_l, other.SS_l);
        Kokkos::deep_copy(SS_v, other.SS_v);
        Kokkos::deep_copy(SS_m, other.SS_m);
        Kokkos::deep_copy(CF_SS, other.CF_SS);
        Kokkos::deep_copy(TM_SS, other.TM_SS);
        Kokkos::deep_copy(VD_SS, other.VD_SS);
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
            Kokkos::resize(SS_l, other.SS_l.extent(0));
            Kokkos::resize(SS_v, other.SS_v.extent(0));
            Kokkos::resize(SS_m, other.SS_m.extent(0));
            Kokkos::resize(CF_SS, other.CF_SS.extent(0));
            Kokkos::resize(TM_SS, other.TM_SS.extent(0));
            Kokkos::resize(VD_SS, other.VD_SS.extent(0));
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
            Kokkos::deep_copy(SS_l, other.SS_l);
            Kokkos::deep_copy(SS_v, other.SS_v);
            Kokkos::deep_copy(SS_m, other.SS_m);
            Kokkos::deep_copy(CF_SS, other.CF_SS);
            Kokkos::deep_copy(TM_SS, other.TM_SS);
            Kokkos::deep_copy(VD_SS, other.VD_SS);
            Kokkos::deep_copy(gk, other.gk);
        }
        return *this;
    }

    // move constructor and assignment for efficiency
    State(State&&) noexcept = default;
    State& operator=(State&&) noexcept = default;
};
