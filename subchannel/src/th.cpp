#include "th.hpp"

void TH::solve_enthalpy(std::vector<double>& h, const std::vector<double>& W_l, double lhr, const Geometry& geom) {

    std::cout << "Solving enthalpy in TH module..." << std::endl;

    // Perform calculations for each axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        h[k] = (W_l[k] * h[k-1] + geom.dz() * lhr) / W_l[k];
    }
}

void TH::solve_pressure(
    std::vector<double>& P,
    const std::vector<double>& W_l,
    const std::vector<double>& rho,
    const std::vector<double>& mu,
    const Geometry& geom
) {

    std::cout << "Solving pressure in TH module..." << std::endl;

    double G; // mass flux
    double Re; // Reynolds number
    double f; // single phase friction factor
    double K; // frictional loss coefficient
    double dP_fric; // single phase frictional pressure drop
    double dP_grav; // single phase gravitational pressure drop

    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        G = W_l[k] / geom.flow_area();
        Re = W_l[k] * geom.hydraulic_diameter() / (geom.flow_area() * mu[k]);

        // calculate frictional pressure drop
        f = 0.1892 * pow(Re, -0.2);
        K = f * geom.dz() / geom.hydraulic_diameter();
        dP_fric = K * G * G / (2.0 * rho[k]);

        // calculate gravitational pressure drop
        dP_grav = rho[k] * 9.81 * geom.dz();

        P[k] = P[k-1] - dP_fric - dP_grav;
    }
}
