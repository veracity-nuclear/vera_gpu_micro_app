#pragma once

#include <iostream>
#include <vector>
#include <memory>

#include "geometry.hpp"

namespace TH {

void solve_enthalpy(std::vector<double>& h, const std::vector<double>& W_l, double lhr, const Geometry& geom);

} // namespace TH
