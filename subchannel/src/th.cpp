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

// void TH::planar(State& state) {

//     size_t k = state.surface_plane;

//     size_t nz = state.geom->naxial(); + 1;

//     double dz = state.geom->dz();
//     double A_f = state.geom->flow_area();
//     double S_ij = state.geom->gap_width();

//     double rho_f = state.fluid->rho_f();
//     double rho_g = state.fluid->rho_g();

//     for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
//         state.alpha[ij][k] = state.alpha[ij][k - 1];
//     }

//     for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {

//         //=======================================================================
//         // * solve for Wm(+), Hm{+), Xf(+), Wml(+), Wmg(+), Hl(+), Rhol(+)
//         //=======================================================================
//         FUNCV_ENTHALPY(wm(ij,k-1), wbarms(ij), hm(ij,k-1), whbars(ij), qz(ij,k), wmg(ij,k-1), wbarvs(ij), gam(ij,k),
//         wml(ij,k), wmg(ij,k), wm(ij,k), hl(ij,k), hm(ij,k), xf(ij,k), rho_l[ij][k]);
//         //=======================================================================
//         // * solve for Alpha(+)
//         //=======================================================================
//         state.alpha[ij][k] = FUNCV_ALPHA(k, ij, wml(ij,k), wmg(ij,k), rho_l[ij][k]);
//         //
//         //   update mixture specific volume and velocity used in axial and transverse pressure drop
//         //       spv   -  specific volume (1/Rhomix) used in the calculation of transverse mixture velocity, mixing and drift terms
//         //       vmn   -  axial mixture velocity (Vmix) used in the calculation of axial and transverse momentum
//         //
//         spv[ij][k] = (1.0 - xf[ij][k])**2 / ((1.0 - state.alpha[ij][k]) * rho_l[ij][k]) + xf[ij][k]**2 / (state.alpha[ij][k] * rho_g + 1.0e-20);
//         vmn[ij][k] = (wml[ij][k] + wmg[ij][k]) * spv[ij][k] / A_f;
//         //
//         //   axial momentum used in acceleration pressure drop calculation
//         //
//         gv[ij][k] = spv[ij][k] * ((wml[ij][k] + wmg[ij][k]) / A_f)**2;

//     }

//     double wvms=0.0;
//     for (auto& surf : state.geom->surfaces) {
//         size_t ns = surf.idx;
//         size_t i = surf.from_node;
//         size_t j = surf.to_node;

//         size_t idonor;
//         if (gk[ns][k] >= 0) {
//             idonor = i;
//         } else {
//             idonor = j;
//         }
//         gkv[ns][k]= gk[ns][k] * vmn[idonor][k];
//         wvms[i] += dz * S_ij * (gkv[ns][k] + mmix[ns] + mdrift[ns]);    // lateral momentum outflow from channel (+)
//         wvms[j] -= dz * S_ij * (gkv[ns][k] + mmix[ns] + mdrift[ns]);    // lateral momentum inflow to channel (-)
//     }

//     for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
//         double deltap = DELTAP_AXIAL(wml(ij,k), wmg(ij,k), state.alpha[ij][k], rho_l[ij][k], xf(ij,k), gv(ij,k), gv(ij,k-1), wvms(ij), xz(k));
//         prz(ij,k)= prz(ij,k-1) - deltap;
//     }
// }

// void TH::FUNCV_ENTHALPY( double wmm, double wbarms, double hmm, double whbars, double qz, double wmgm,
//     double wbarvs, double gam, double wmlp, double wmgp, double wmp, double hlp, double hmp, double xfp, double rholp) {

//     double wmg_local;
//     double xf_local;
//     double hl_local;

//     double A_f = state.geom->flow_area();
//     double A_wall = state.geom->heated_perimeter() * state.geom->dz();

//     //=======================================================================
//     // * solve for Wm(+) and Hm(+), this will be true regardless based on
//     //   mass and energy balance
//     //=======================================================================
//     wmp = wmm - wbarms;
//     hmp = (wmm * hmm + qz * A_wall - whbars) / wmp;
//     //=======================================================================
//     // * solve for Xf(+) and Hl(+), noting Hl(+) cannot exceed saturation
//     //
//     //   1) for Hl(+) < state.fluid->h_f(), calculate Xf(+) from Wv(+) and Gam(+)
//     //   2) otherwise, Hl(+) = state.fluid->h_f() and Xf(+) calculated directly from Hm(+)
//     //=======================================================================
//     wmg_local = wmgm - wbarvs + gam * A_f;
//     wmg_local = max(0.0, wmg_local);
//     xf_local =  wmg_local / wmp;
//     hl_local = (hmp - xf_local * state.fluid->h_g()) / (1.0 - xf_local);
//     if (hl_local >= state.fluid->h_f()) {
//         xf_local = (hmp - state.fluid->h_f()) / (state.fluid->h_g() - state.fluid->h_f());
//         hlp = state.fluid->h_f();
//     } else {
//         xfp = xf_local;
//         hlp = hl_local;
//     }
//     //=======================================================================
//     // * given Xf(+), calculate updated Wl(+), Wg(+), and Rhol(+)
//     //=======================================================================
//     wmlp = (1.0 - xfp)*wmp;
//     wmgp = xfp*wmp;
//     rholp = state.fluid->rho(hlp);
// }

// double TH::FUNCV_ALPHA(k, ij, wmlp, wmgp, rholp) {
//     USE UTIL_MOD, only  : C0SC, VGJ
//     USE FLUX_MODB, only : A_f, max_inners, rhog, eps, atol
//     integer          ::               k
//     integer          ::               ij
//     double              wmlp
//     double              wmgp
//     double              rholp
//     double              alphzp
//     double              gl
//     double              gv
//     double              alphx
//     double              fv0
//     double              fv1
//     double              dfda
//     double              delta_a
//     double              alphx1
//     integer::                         its
//     if (wmlp < eps) then
//         write(*,*) 'k: ',k,' ij0: ',ij,' Negative liquid flow: ',wmlp,wmgp
//         stop
//     endif
//     gl = wmlp/A_f
//     gv = wmgp/A_f
//     alphx = alphzp
//     do its = 1, max_inners
//         fv0 = alphx*C0SC(alphx)*(rhog/rholp*gl + gv) + alphx*rhog*VGJ(alphx) - gv
//         if (abs(fv0) < eps) exit
//         alphx1 = alphx + atol
//         fv1 = alphx1*C0SC(alphx1)*(rhog/rholp*gl + gv) + alphx1*rhog*VGJ(alphx1) - gv
//         dfda= (fv1 - fv0)/atol
//         delta_a = -fv0/dfda
//         delta_a =min(delta_a, 0.1d0)
//         delta_a =max(delta_a,-0.1d0)
//         alphx = alphx + delta_a
//         alphx = min(alphx,(1.d0 - eps))
//         alphx = max(alphx,0.d0)
//     end do
//     alphzp = alphx
// }

// double TH::DELTAP_AXIAL(wmlp, wmgp, alphzp, rholp, xfp, gvp, gvm, wvms, xzp) {
//     USE FLUX_MODB, only   : rhog, vfff, viscf0, bstar, gamasq1, dzth, dh, A_f
//     USE PROPS_MODB, only  : ggrav
//     double wmlp;
//     double wmgp;
//     double alphzp;
//     double rholp;
//     double xfp;
//     double gvp;
//     double gvm;
//     double wvms;
//     double xzp;
//     double deltap;
//     double gbara;
//     double b;
//     double xfp9;
//     double thomo;
//     double tchsm;
//     double re;
//     double kbar;
//     double dP1;
//     double dP2;
//     double dP3;
//     double dP4;
//     double dP0e;
//     double dP0c;
//     gbara=abs(wmlp+wmgp)/A_f
//     //-----------------------------------------------------------------------
//     // * calculate homogeneous 2-phase multiplier
//     //-----------------------------------------------------------------------
//     thomo=1.d0+xfp*(rholp/rhog - 1.d0)
//     //-----------------------------------------------------------------------
//     // * calculate Chisholm-Baroczy 2-phase multiplier
//     //-----------------------------------------------------------------------
//     if (xfp < 1.d0) then
//         b=bstar/dsqrt(gbara)
//         xfp9=xfp**.9d0
//         tchsm=1.d0+gamasq1*(b*xfp9*(1.d0-xfp)**0.9d0+xfp9*xfp9)
//     else
//         tchsm=1.d0+gamasq1
//     endif
//     //-----------------------------------------------------------------------
//     // * calculate Reynold's number using node average mass flux and
//     //   saturated liquid dynamic viscosity; calculate effective loss
//     //   coefficient
//     //-----------------------------------------------------------------------
//     re=gbara*dh/viscf0
//     kbar=0.1892d0*re**(-0.2d0)*dzth/dh*tchsm + xzp*thomo
//     //
//     // ... dP1 is based on the NEW Wliq, Wvap, and Xf

//     dP1= 0.5d0*kbar*gbara*gbara*vfff
//     //
//     // ... dP2 is the net momentum outflow based on the PREVIOUS outer iteration values of Gk*V, mixing momentum, and drift momentum
//     //
//     dP2= wvms/A_f
//     //
//     // ... dP3 is based on the NEW Hliq and Alpha
//     //
//     dP3= ggrav*dzth*((1.d0-alphzp)*rholp + alphzp*rhog)
//     //
//     // ... dP4 is based on the NEW G*V
//     //
//     dP4= gvp - gvm
//     dP0e= 0.d0
//     dP0c= 0.d0
//     deltap= dP0e + dP0c + dP1 + dP2 + dP3 + dP4
// }

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

        double Qflux_boil; // boiling heat flux [W/m], Eq. 51 from ANTS Theory
        if (state.h_l[i][k] < state.fluid->h_f()) {
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
            Qflux_boil = Qflux_wall;
            state.evap[i][k_node] = P_H * Qflux_boil / state.fluid->h_fg();
        }
        state.qz[i][k_node] = Qflux_boil; // heat flux [W/m^2]
    }
}

void TH::solve_mixing(State& state) {

    const double Thetam = 5.0; // constant set equal to 5.0 for BWR applications, from ANTS Theory

    size_t k = state.surface_plane;  // closure relations use lagging edge values
    size_t k_node = state.node_plane;

    Vector2D rho_l = state.fluid->rho(state.h_l);
    Vector2D spv = state.fluid->mu(state.h_l);
    double rhof = state.fluid->rho_f();
    double rhog = state.fluid->rho_g();

    double A_f = state.geom->flow_area(); // flow area [m^2]
    double D_rod = state.geom->heated_perimeter() / M_PI; // rod diameter [m], assuming square array
    double D_h = state.geom->hydraulic_diameter(); // hydraulic diameter [m]
    double S_ij = state.geom->gap_width(); // gap width between subchannels [m]

    // precalculate the two-phase multipliers on a subchannel basis
    Vector1D gbar0(state.geom->nchannels(), 0.0);
    Vector1D reyn0(state.geom->nchannels(), 0.0);
    Vector1D Theta(state.geom->nchannels(), 0.0);
    for (size_t ij = 0; ij < state.geom->nchannels(); ++ij) {
        double viscmi = state.X[ij][k] / state.fluid->mu_g() + (1.0 - state.X[ij][k]) / state.fluid->mu_f();
        gbar0[ij] = (state.W_l[ij][k] + state.W_v[ij][k]) / A_f;
        reyn0[ij] = gbar0[ij] * D_h / viscmi;

        double Xmm = (0.4 * std::sqrt(rhof * (rhof - rhog) * g * D_h) / gbar0[ij] + 0.6) / (std::sqrt(rhof / rhog) + 0.6);
        double X0m = 0.57 * std::pow(reyn0[ij], 0.0417);
        double Xfm = state.X[ij][k] / Xmm;
        if (state.X[ij][k] < Xmm) {
            Theta[ij] = 1.0 + (Thetam - 1.0) * Xfm;
        } else {
            Theta[ij] = 1.0 + (Thetam - 1.0) * (1.0 - X0m) / (Xfm - X0m);
        }
    }

    // loop over surfaces
    for (auto& surf : state.geom->surfaces) {
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;

        double G_m_i = state.W_m(i, k) / A_f;
        double G_m_j = state.W_m(j, k) / A_f;
        double G_m_avg = 0.5 * (G_m_i + G_m_j);
        double rho_l_i = rho_l[i][k];
        double rho_l_j = rho_l[j][k];
        double h_l_i = state.h_l[i][k];
        double h_l_j = state.h_l[j][k];
        double h_l_avg = 0.5 * (h_l_i + h_l_j);
        double alpha_i = state.alpha[i][k];
        double alpha_j = state.alpha[j][k];
        double V_l_i = __liquid_velocity(state.W_l[i][k], A_f, alpha_i, rho_l_i);
        double V_l_j = __liquid_velocity(state.W_l[j][k], A_f, alpha_j, rho_l_j);
        double V_v_i = __vapor_velocity(state.W_v[i][k], A_f, alpha_i, rhog);
        double V_v_j = __vapor_velocity(state.W_v[j][k], A_f, alpha_j, rhog);
        double Re = __Reynolds(G_m_avg, D_h, state.fluid->mu(h_l_avg));
        double X_bar = __quality_avg(G_m_i, G_m_j);
        double tp_mult = 0.5 * (Theta[i] + Theta[j]);
        double lambda = 0.0058 * S_ij / D_rod; // Eq. 46 from ANTS Theory
        double reynbar = 0.5 * (reyn0[i] + reyn0[j]);
        double spbar = 0.5 * (spv[i][k] + spv[j][k]);
        double eddy_V;
        if (reyn0[i] < reyn0[j]) {
            eddy_V= 0.5 * lambda * std::pow(reynbar, -0.1) * (1.0 + std::pow(D_h / D_h, 1.5)) * (D_h / D_rod) * gbar0[i] * spbar;
        } else {
            eddy_V= 0.5 * lambda * std::pow(reynbar, -0.1) * (1.0 + std::pow(D_h / D_h, 1.5)) * (D_h / D_rod) * gbar0[j] * spbar;
        }

        // turbulent mixing liquid mass transfer
        state.G_l_tm[ns] = eddy_V * tp_mult * (
            (1 - alpha_i) * rho_l_i - (1 - alpha_j) * rho_l_j
        ); // Eq. 37 from ANTS Theory

        // turbulent mixing vapor mass transfer
        state.G_v_tm[ns] = eddy_V * tp_mult * state.fluid->rho_g() * (alpha_i - alpha_j); // Eq. 38 from ANTS Theory

        // turbulent mixing energy transfer
        state.Q_m_tm[ns] = eddy_V * tp_mult * (
            (1 - alpha_i) * rho_l_i * h_l_i + alpha_i * state.fluid->rho_g() * state.fluid->h_g()
            - (1 - alpha_j) * rho_l_j * h_l_j - alpha_j * state.fluid->rho_g() * state.fluid->h_g()
        ); // Eq. 39 from ANTS Theory

        // turbulent mixing momentum transfer
        state.M_m_tm[ns] = eddy_V * tp_mult * (G_m_i - G_m_j); // Eq. 40 from ANTS Theory

        // void drift liquid mass transfer
        state.G_l_vd[ns] = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i + alpha_j * rho_l_j
        ); // Eq. 42 from ANTS Theory

        // void drift vapor mass transfer
        state.G_v_vd[ns] = -eddy_V * tp_mult * X_bar * (
            alpha_i + alpha_j
        ) * state.fluid->rho_g(); // Eq. 41 from ANTS Theory

        // void drift energy transfer
        state.Q_m_vd[ns] = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i * h_l_i + alpha_j * rho_l_j * h_l_j
            - (alpha_i + alpha_j) * state.fluid->rho_g() * state.fluid->h_g()
        ); // Eq. 43 from ANTS Theory

        // void drift momentum transfer
        state.M_m_vd[ns] = eddy_V * tp_mult * X_bar * (
            alpha_i * rho_l_i * V_l_i + alpha_j * rho_l_j * V_l_j
            - (alpha_i * V_v_i + alpha_j * V_v_j) * state.fluid->rho_g()
        ); // Eq. 44 from ANTS Theory

        state.gk[ns][k_node] = state.G_l_tm[ns] + state.G_v_tm[ns] + state.G_l_vd[ns] + state.G_v_vd[ns];
    }
}

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
    Vector2D &gk = state.gk; // total transverse mass flux
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
        if (gk[ns][k] > 0) idonor = surf.from_node;
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

    // loop over transverse surfaces to add source terms to flow rates
    Vector1D SS_l(state.geom->nchannels());
    Vector1D SS_v(state.geom->nchannels());
    for (auto& surf : state.geom->surfaces) {
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;

        SS_l[i] += state.geom->gap_width() * (state.G_l_tm[ns] + state.G_l_vd[ns]);
        SS_l[j] -= state.geom->gap_width() * (state.G_l_tm[ns] + state.G_l_vd[ns]);

        SS_v[i] += state.geom->gap_width() * (state.G_v_tm[ns] + state.G_v_vd[ns]);
        SS_v[j] -= state.geom->gap_width() * (state.G_v_tm[ns] + state.G_v_vd[ns]);
    }

    if (k_node == 3) {
        for (size_t i = 0; i < state.geom->nchannels(); ++i) {
            std::cout << "Channel " << i << " SS_l: " << SS_l[i] << " SS_v: " << SS_v[i] << std::endl;
        }
    }

    for (size_t i = 0; i < state.geom->nchannels(); ++i) {

        state.W_l[i][k] = state.W_l[i][k-1] - state.geom->dz() * (state.evap[i][k_node] + SS_l[i]); // Eq. 61 from ANTS Theory

        // throw error if liquid flow rate becomes negative and add debug info
        if (state.W_l[i][k] < 0) {
            throw std::runtime_error("Error: Liquid flow rate has become negative in channel " + std::to_string(i) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "Previous W_l: " + std::to_string(state.W_l[i][k-1]) + "\n"
                "Evaporation term: " + std::to_string(state.evap[i][k_node]) + "\n"
                "SS_l: " + std::to_string(SS_l[i]) + "\n");
        }

        state.W_v[i][k] = state.W_v[i][k-1] + state.geom->dz() * (state.evap[i][k_node] - SS_v[i]); // Eq. 62 from ANTS Theory

        // throw error if vapor flow rate becomes negative and add debug info
        if (state.W_v[i][k] < 0) {
            throw std::runtime_error("Error: Vapor flow rate has become negative in channel " + std::to_string(i) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "Previous W_v: " + std::to_string(state.W_v[i][k-1]) + "\n"
                "Evaporation term: " + std::to_string(state.evap[i][k_node]) + "\n"
                "SS_v: " + std::to_string(SS_v[i]) + "\n");
        }
    }

    // std::cout << " ----- PLANE " << k << " ----- " << std::endl;

    // std::cout << "Liq. Mass Flow Rates:" << std::endl;
    // for (size_t i = 0; i < state.geom->nchannels(); ++i) {
    //     std::cout << std::setw(12) << state.W_l[i][k];
    //     if ((i + 1) % 3 == 0) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // std::cout << "Vap. Mass Flow Rates:" << std::endl;
    // for (size_t i = 0; i < state.geom->nchannels(); ++i) {
    //     std::cout << std::setw(12) << state.W_v[i][k];
    //     if ((i + 1) % 3 == 0) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // std::cout << "Total Mass Flow Rates:" << std::endl;
    // for (size_t i = 0; i < state.geom->nchannels(); ++i) {
    //     std::cout << std::setw(12) << (state.W_l[i][k] + state.W_v[i][k]);
    //     if ((i + 1) % 3 == 0) {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;
}

void TH::solve_enthalpy(State& state) {
    size_t k = state.surface_plane;
    size_t k_node = state.node_plane;

    // loop over transverse surfaces to add source terms to mixture enthalpy
    Vector1D SS_m(state.geom->nchannels());
    for (auto& surf : state.geom->surfaces) {
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;

        SS_m[i] += state.geom->gap_width() * (state.Q_m_tm[ns] + state.Q_m_vd[ns]);
        SS_m[j] -= state.geom->gap_width() * (state.Q_m_tm[ns] + state.Q_m_vd[ns]);
    }

    for (size_t i = 0; i < state.geom->nchannels(); ++i) {

        state.h_l[i][k] = (
            (state.W_v[i][k-1] - state.W_v[i][k]) * state.fluid->h_g()
            + state.W_l[i][k-1] * state.h_l[i][k-1] + state.geom->dz() * state.lhr[i][k_node]
            - state.geom->dz() * SS_m[i]
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

        double Re_g = __Reynolds(state.W_v[i][k] / A, D_h, mu_v); // local vapor Reynolds number
        double Re_f = __Reynolds(state.W_l[i][k] / A, D_h, mu_l); // local liquid Reynolds number
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

        state.alpha[i][k] = bisection(f, 0.0, 1.0, tol, 100); // solve for void fraction using bisection method
    }
}

void TH::solve_quality(State& state) {
    size_t k = state.surface_plane;
    for (size_t i = 0; i < state.geom->nchannels(); ++i) {
        double G_v = state.W_v[i][k] / state.geom->flow_area(); // vapor mass flux, Eq. 8 from ANTS Theory
        double G_l = state.W_l[i][k] / state.geom->flow_area(); // liquid mass flux, Eq. 9 from ANTS Theory
        state.X[i][k] = G_v / (G_v + G_l); // Eq. 17 from ANTS Theory

        // throw error if quality becomes negative and add debug info
        if (state.X[i][k] < 0.0) {
            throw std::runtime_error("Error: Quality has become negative in channel " + std::to_string(i) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "G_v: " + std::to_string(G_v) + "\n"
                "G_l: " + std::to_string(G_l) + "\n");
        }
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

    // loop over transverse surfaces to add source terms to pressure drops
    Vector1D TM_SS(state.geom->nchannels()); // sum of turbulent mixing liquid momentum exchange terms [Pa]
    Vector1D VD_SS(state.geom->nchannels()); // sum of void drift liquid momentum exchange terms [Pa]
    for (auto& surf : state.geom->surfaces) {
        size_t ns = surf.idx;
        size_t i = surf.from_node;
        size_t j = surf.to_node;

        TM_SS[i] += state.geom->gap_width() * state.M_m_tm[ns];
        TM_SS[j] -= state.geom->gap_width() * state.M_m_tm[ns];

        VD_SS[i] += state.geom->gap_width() * state.M_m_vd[ns];
        VD_SS[j] -= state.geom->gap_width() * state.M_m_vd[ns];
    }

    for (size_t i = 0; i < state.geom->nchannels(); ++i) {

        // ----- two-phase acceleration pressure drop -----
        double dP_accel = (state.W_m(i, k) * state.V_m(i, k) - state.W_m(i, k-1) * state.V_m(i, k-1)) / A_f;

        // ----- two-phase frictional pressure drop -----
        // mass flux
        double G = (state.W_l[i][k] + state.W_v[i][k]) / A_f;
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

        // throw error if phi2_ch is NaN and add debug info
        if (std::isnan(phi2_ch)) {
            throw std::runtime_error("Error: Two-phase multiplier has become NaN in channel " + std::to_string(i) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "gamma: " + std::to_string(gamma) + "\n"
                "b: " + std::to_string(b) + "\n"
                "X: " + std::to_string(state.X[i][k]) + "\n");
        }

        // two-phase wall shear pressure drop, Eq. 29 from ANTS Theory
        double dP_wall_shear = K * G * G / (2.0 * rho[i][k]) * phi2_ch;

        // throw error if dP_wall_shear is NaN and add debug info
        if (std::isnan(dP_wall_shear)) {
            throw std::runtime_error("Error: Two-phase wall shear pressure drop has become NaN in channel " + std::to_string(i) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "K: " + std::to_string(K) + "\n"
                "G: " + std::to_string(G) + "\n"
                "rho: " + std::to_string(rho[i][k]) + "\n"
                "phi2_ch: " + std::to_string(phi2_ch) + "\n"
            );
        }

        // form loss coefficient (no form losses in this simple model)
        double K_loss = 0.0;

        // two-phase multiplier for form losses (homogeneous), Eq. 35 from ANTS Theory
        double phi2_hom = 1.0 + state.X[i][k] * (rho[i][k] / state.fluid->rho_g() - 1.0);

        // two-phase geometry form loss pressure drop, Eq. 36 from ANTS Theory
        double dP_form = K_loss * G * G / (2.0 * rho[i][k]) * phi2_hom;

        // two-phase frictional pressure drop, Eq. 36 from ANTS Theory
        double dP_tpfric = dP_wall_shear + dP_form;

        // throw error if dP_tpfric is NaN and add debug info
        if (std::isnan(dP_tpfric)) {
            throw std::runtime_error("Error: Two-phase frictional pressure drop has become NaN in channel " + std::to_string(i) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "dP_wall_shear: " + std::to_string(dP_wall_shear) + "\n"
                "dP_form: " + std::to_string(dP_form) + "\n");
        }

        // ----- two-phase gravitational pressure drop -----
        double dP_grav = rho[i][k] * g * dz;

        // ----- momentum exchange due to pressure-directed crossflow, turbulent mixing, and void drift -----
        double dP_CF = 0.0; // implement to complete Issue #73
        double dP_TM = dz / A_f * TM_SS[i];
        double dP_VD = dz / A_f * VD_SS[i];
        double dP_momexch = dP_CF + dP_TM + dP_VD;

        // ----- total pressure drop over this axial plane -----
        double dP_total = dP_accel + dP_tpfric + dP_grav + dP_momexch; // Eq. 65 from ANTS Theory
        state.P[i][k] = state.P[i][k-1] - dP_total;

        // throw error if pressure goes NaN and add debug info (dP_accel, dP_tpfric, dP_grav, dP_momexch, dP_total)
        if (std::isnan(state.P[i][k])) {
            throw std::runtime_error("Error: Pressure has become NaN in channel " + std::to_string(i) + " at plane " + std::to_string(k) + ".\n"
                "Debug Info:\n"
                "dP_accel: " + std::to_string(dP_accel) + "\n"
                "dP_tpfric: " + std::to_string(dP_tpfric) + "\n"
                "dP_grav: " + std::to_string(dP_grav) + "\n"
                "dP_momexch: " + std::to_string(dP_momexch) + "\n"
                "dP_total: " + std::to_string(dP_total) + "\n");
        }
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
