#pragma once

#include "vectors.hpp"

struct State {
    size_t surface_plane = 0;       // current surface axial plane being solved
    size_t node_plane = 0;          // current node axial plane being solved
    std::shared_ptr<Water> fluid;   // reference to fluid properties
    std::shared_ptr<Geometry> geom; // reference to geometry

    Vector3D h_l;       // liquid enthalpy
    Vector3D W_l;       // liquid mass flow rate
    Vector3D W_v;       // vapor mass flow rate
    Vector3D P;         // pressure
    Vector3D alpha;     // void fraction
    Vector3D X;         // quality
    Vector3D lhr;       // linear heat rate
    Vector3D evap;      // evaporation term [kg/m/s]

    Vector4D G_l_tm;    // turbulent mixing liquid mass transfer [kg/m^2/s]
    Vector4D G_v_tm;    // turbulent mixing vapor mass transfer [kg/m^2/s]
    Vector4D Q_m_tm;    // turbulent mixing energy transfer [W/m^2]
    Vector4D M_m_tm;    // turbulent mixing momentum transfer [Pa]

    Vector4D G_l_vd;    // void drift liquid mass transfer [kg/m^2/s]
    Vector4D G_v_vd;    // void drift vapor mass transfer [kg/m^2/s]
    Vector4D Q_m_vd;    // void drift energy transfer [W/m^2]
    Vector4D M_m_vd;    // void drift momentum transfer [Pa]

    // mixture enthalpy
    Vector3D h_m() const {
        if (W_l.size() != W_v.size() || h_l.size() != W_l.size()) {
            throw std::length_error("State::h_m(): h_l, W_l, and W_v have different sizes");
        }
        Vector3D out(h_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            for (std::size_t j = 0; j < out[i].size(); ++j) {
                for (std::size_t k = 0; k < out[i][j].size(); ++k) {
                    double X_ijk = X[i][j][k];
                    out[i][j][k] = h_m(i, j, k);
                }
            }
        }
        return out;
    }

    double h_m(size_t i, size_t j, size_t k) const {
        return (1 - X[i][j][k]) * h_l[i][j][k] + X[i][j][k] * fluid->h_g();
    }

    // mixture mass flow rate
    Vector3D W_m() const {
        if (W_l.size() != W_v.size()) {
            throw std::length_error("State::W_m(): W_l and W_v have different sizes");
        }
        Vector3D out(W_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            for (std::size_t j = 0; j < out[i].size(); ++j) {
                for (std::size_t k = 0; k < out[i][j].size(); ++k) {
                    out[i][j][k] = W_m(i, j, k);
                }
            }
        }
        return out;
    }

    double W_m(size_t i, size_t j, size_t k) const {
        return W_l[i][j][k] + W_v[i][j][k];
    }

    // mixture velocity
    Vector3D V_m() const {
        Vector3D out(W_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            for (std::size_t j = 0; j < out[i].size(); ++j) {
                for (std::size_t k = 0; k < out[i][j].size(); ++k) {
                    out[i][j][k] = V_m(i, j, k);
                }
            }
        }
        return out;
    }

    double V_m(size_t i, size_t j, size_t k) const {
        Vector3D rho = fluid->rho(h_l);
        double A_f = geom->flow_area();
        double v_m;
        if (alpha[i][j][k] < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha[i][j][k] > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X[i][j][k]) * (1.0 - X[i][j][k]) / ((1.0 - alpha[i][j][k]) * rho[i][j][k]) + X[i][j][k] * X[i][j][k] / (alpha[i][j][k] * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m * W_m(i, j, k) / A_f; // mixture velocity, Eq. 15 from ANTS Theory
    }

    // mixture specific volume
    Vector3D nu_m() const {
        Vector3D out(W_l.size());
        for (std::size_t i = 0; i < out.size(); ++i) {
            for (std::size_t j = 0; j < out[i].size(); ++j) {
                for (std::size_t k = 0; k < out[i][j].size(); ++k) {
                    out[i][j][k] = nu_m(i, j, k);
                }
            }
        }
        return out;
    }

    double nu_m(size_t i, size_t j, size_t k) const {
        Vector3D rho = fluid->rho(h_l);
        double v_m;
        if (alpha[i][j][k] < 1e-6) {
            v_m = 1.0 / fluid->rho_f(); // Eq. 16 from ANTS Theory (Simplified with X=0, alpha=0)
        } else if (alpha[i][j][k] > 1.0 - 1e-6) {
            v_m = 1.0 / fluid->rho_g(); // Eq. 16 from ANTS Theory (Simplified with X=1, alpha=1)
        } else {
            v_m = (1.0 - X[i][j][k]) * (1.0 - X[i][j][k]) / ((1.0 - alpha[i][j][k]) * rho[i][j][k]) + X[i][j][k] * X[i][j][k] / (alpha[i][j][k] * fluid->rho_g()); // Eq. 16 from ANTS Theory
        }
        return v_m; // mixture specific volume
    }

    double G_m_cf(size_t i, size_t j, size_t ns) const {

        if (ns == geom->boundary) {
            return 0.0; // no crossflow at boundary surfaces
        }

        size_t from_i, from_j;
        std::tie(from_i, from_j) = geom->get_ij(geom->surfaces[ns].from_node);
        size_t from_ns = geom->local_surf_index(geom->surfaces[ns], geom->surfaces[ns].from_node);

        // Update surface cross-flow mass fluxes in the to node
        if (from_i == i && from_j == j) {
            // return -1.0;
            return -geom->surfaces[ns].G;
        } else {
            // return 1.0;
            return geom->surfaces[ns].G;
        }
    }

    // double G_l_cf(size_t i, size_t j, size_t ns) const {

    //     if (ns == geom->boundary) {
    //         return 0.0; // no crossflow at boundary surfaces
    //     }

    //     size_t from_i, from_j;
    //     std::tie(from_i, from_j) = geom->get_ij(geom->surfaces[ns].from_node);
    //     size_t from_ns = geom->local_surf_index(geom->surfaces[ns], geom->surfaces[ns].from_node);

    //     // Get donor quality from appropriate subchannel surface
    //     double X_donor = X[from_i][from_j][surface_plane - 1];

    //     return G_m_cf(i, j, ns) * (1.0 - X_donor);
    // }

    // double G_v_cf(size_t i, size_t j, size_t ns) const {

    //     if (ns == geom->boundary) {
    //         return 0.0; // no crossflow at boundary surfaces
    //     }

    //     size_t from_i, from_j;
    //     std::tie(from_i, from_j) = geom->get_ij(geom->surfaces[ns].from_node);
    //     size_t from_ns = geom->local_surf_index(geom->surfaces[ns], geom->surfaces[ns].from_node);

    //     // Get donor quality from appropriate subchannel surface
    //     double X_donor = X[from_i][from_j][surface_plane - 1];

    //     return G_m_cf(i, j, ns) * X_donor;
    // }
};
