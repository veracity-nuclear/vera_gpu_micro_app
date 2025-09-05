#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "vectors.hpp"
#include "geometry.hpp"
#include "materials.hpp"

namespace TH {

void solve_enthalpy(Vector1D& h, const Vector1D& W_l, double lhr, const Geometry& geom);
void solve_pressure(Vector1D& P, const Vector1D& W_l, const Vector1D& rho, const Vector1D& mu, const Geometry& geom, const Water& fluid);

} // namespace TH
