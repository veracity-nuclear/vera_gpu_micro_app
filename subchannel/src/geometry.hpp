#pragma once

#include <cstddef>

class Geometry {
public:
    Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, size_t naxial);
    ~Geometry() = default;

    double height() const { return H; }
    double flow_area() const { return Af; }
    double hydraulic_diameter() const { return Dh; }
    double gap_width() const { return gap_W; }
    double heated_perimeter() const { return Af / dz(); }
    double dz() const { return H / nz; }
    size_t naxial() const { return nz; }

private:
    double H;
    double Af;
    double Dh;
    double gap_W;
    size_t nz;
};
