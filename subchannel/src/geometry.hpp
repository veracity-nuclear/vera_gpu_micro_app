#pragma once

#include <cstddef>

class Geometry {
public:
    Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, size_t nx, size_t ny, size_t naxial);
    ~Geometry() = default;

    double height() const { return H; }
    double flow_area() const { return Af; }
    double hydraulic_diameter() const { return Dh; }
    double gap_width() const { return gap_W; }
    double heated_perimeter() const { return 4.0 * Af / Dh; }
    double dz() const { return H / _nz; }
    size_t nx() const { return _nx; }
    size_t ny() const { return _ny; }
    size_t naxial() const { return _nz; }

private:
    double H;
    double Af;
    double Dh;
    double gap_W;
    size_t _nx, _ny, _nz;
};
