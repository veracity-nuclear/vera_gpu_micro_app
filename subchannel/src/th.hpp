#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "geometry.hpp"
#include "materials.hpp"

namespace TH {

void solve_enthalpy(std::vector<double>& h, const std::vector<double>& W_l, double lhr, const Geometry& geom);
void solve_pressure(
    std::vector<double>& P,
    const std::vector<double>& W_l,
    const std::vector<double>& rho,
    const std::vector<double>& mu,
    const Geometry& geom
);

} // namespace TH
