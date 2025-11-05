#pragma once

#include <iostream>
#include <cstddef>
#include <utility>
#include <limits>
#include <vector>


struct Surface {
    size_t idx;         // surface index
    size_t from_node;   // upstream node index
    size_t to_node;     // downstream node index
    double G;           // mixture cross-flow mass flux [kg/m^2-s]
};


class Geometry {
public:
    Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, double length, size_t nx, size_t ny, size_t naxial);
    ~Geometry() = default;

    std::vector<Surface> surfaces; // list of surfaces between subchannels
    const size_t boundary = std::numeric_limits<size_t>::max(); // for a neighbor that is a boundary
    double height() const { return H; }
    double flow_area() const { return Af; }
    double hydraulic_diameter() const { return Dh; }
    double gap_width() const { return gap_W; }
    double heated_perimeter() const { return 4.0 * Af / Dh; }
    double aspect_ratio() const { return gap_W / l; }
    double dz() const { return H / _nz; }
    size_t nx() const { return _nx; }
    size_t ny() const { return _ny; }
    size_t naxial() const { return _nz; }
    size_t nsurfaces() const;
    size_t global_surf_index(size_t i, size_t j, size_t ns) const;
    size_t local_surf_index(Surface surf, size_t node_idx) const;
    std::pair<size_t, size_t> get_ij(size_t idx) const;
    std::pair<size_t, size_t> get_donor_ij(size_t ns) const;
    std::pair<size_t, size_t> get_neighbor_ij(size_t i, size_t j, size_t ns) const;

private:
    double H;       // height of the subchannel
    double Af;      // flow area
    double Dh;      // hydraulic diameter
    double gap_W;   // gap width between subchannels
    double l;       // length of axial momentum cell
    size_t _nx, _ny, _nz;  // number of subchannels in x, y directions, and number of axial cells
};
