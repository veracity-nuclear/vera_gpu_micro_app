#include "th.hpp"

void TH::solve_enthalpy(std::vector<double>& h, const std::vector<double>& W_l, double lhr, const Geometry& geom) {

    std::cout << "Solving enthalpy in TH module..." << std::endl;

    // Perform calculations for each axial plane
    for (size_t k = 1; k < geom.naxial() + 1; ++k) {
        h[k] = (W_l[k] * h[k-1] + geom.dz() * lhr) / W_l[k];
    }
}
