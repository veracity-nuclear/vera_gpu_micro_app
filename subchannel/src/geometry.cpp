#include "geometry.hpp"

Geometry::Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, double length, size_t nx, size_t ny, size_t naxial)
    : H(height), Af(flow_area), Dh(hydraulic_diameter), gap_W(gap_width), l(length), _nx(nx), _ny(ny), _nz(naxial) {

        // create all surfaces in between subchannels for any axial plane
        for (size_t j = 0; j < _ny; ++j) {
            for (size_t i = 0; i < _nx; ++i) {
                // vertical surfaces (between subchannels in x-direction)
                if (i < _nx - 1) {
                    Surface surf;
                    surf.idx = surfaces.size();
                    surf.from_node = j * _nx + i;
                    surf.to_node = j * _nx + (i + 1);
                    surfaces.push_back(surf);
                }
            }
        }
        for (size_t j = 0; j < _ny; ++j) {
            for (size_t i = 0; i < _nx; ++i) {
                // horizontal surfaces (between subchannels in y-direction)
                if (j < _ny - 1) {
                    Surface surf;
                    surf.idx = surfaces.size();
                    surf.from_node = j * _nx + i;
                    surf.to_node = (j + 1) * _nx + i;
                    surfaces.push_back(surf);
                }
            }
        }
    }

size_t Geometry::nsurfaces() const {
    // Total number of internal surfaces in the subchannel grid
    return (_nx - 1) * _ny + (_ny - 1) * _nx;
}

size_t Geometry::global_surf_index(size_t i, size_t ns) const {
    // Calculate global surface index based on subchannel index (i) and surface number (ns)
    // Surface numbering: 0 = west, 1 = east, 2 = north, 3 = south
    size_t nsurf_x = _nx - 1;
    size_t nsurf_y = _ny - 1;
    if (ns == 0) {
        if (i % _nx == 0) return boundary; // left boundary
        return i / _nx * nsurf_x + i % _nx - 1; // west surface
    } else if (ns == 1) {
        if (i % _nx == _nx - 1) return boundary; // right boundary
        return i / _nx * nsurf_x + i % _nx; // east surface
    } else if (ns == 2) {
        if (i / _nx == 0) return boundary; // top boundary
        return nsurf_x * _ny + (i / _nx - 1) * _nx + i % _nx; // north surface
    } else if (ns == 3) {
        if (i / _nx == _ny - 1) return boundary; // bottom boundary
        return nsurf_x * _ny + i / _nx * _nx + i % _nx; // south surface
    }
    return boundary;
}
