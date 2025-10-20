#include "geometry.hpp"

Geometry::Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, size_t nx, size_t ny, size_t naxial)
    : H(height), Af(flow_area), Dh(hydraulic_diameter), gap_W(gap_width), _nx(nx), _ny(ny), _nz(naxial) {}