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
                    surf.G = 0.0;
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
                    surf.G = 0.0;
                    surfaces.push_back(surf);
                }
            }
        }
    }

size_t Geometry::nsurfaces() const {
    // Total number of internal surfaces in the subchannel grid
    return (_nx - 1) * _ny + (_ny - 1) * _nx;
}

size_t Geometry::global_surf_index(size_t i, size_t j, size_t ns) const {
    // Calculate global surface index based on subchannel indices (i, j) and surface number ns
    // Surface numbering: 0 = west, 1 = east, 2 = north, 3 = south
    size_t nsurf_x = _nx - 1;
    size_t nsurf_y = _ny - 1;
    if (ns == 0) {
        if (i == 0) return boundary; // left boundary
        return j * nsurf_x + i - 1; // west surface
    } else if (ns == 1) {
        if (i == _nx - 1) return boundary; // right boundary
        return j * nsurf_x + i; // east surface
    } else if (ns == 2) {
        if (j == 0) return boundary; // top boundary
        return nsurf_x * _ny + (j - 1) * _nx + i; // north surface
    } else if (ns == 3) {
        if (j == _ny - 1) return boundary; // bottom boundary
        return nsurf_x * _ny + j * _nx + i; // south surface
    }
    return boundary;
}

size_t Geometry::local_surf_index(Surface surf, size_t node_idx) const {
    // Calculate local surface index based on node index and the surface's from_node and to_node
    size_t i, j;
    std::tie(i, j) = get_ij(node_idx);

    size_t neigh_idx;
    if (node_idx == surf.from_node) {
        neigh_idx = surf.to_node;
    } else if (node_idx == surf.to_node) {
        neigh_idx = surf.from_node;
    }

    size_t ns = boundary;
    if (neigh_idx == node_idx - 1) {
        ns = 0; // west
    } else if (neigh_idx == node_idx + 1) {
        ns = 1; // east
    } else if (neigh_idx == node_idx - _nx) {
        ns = 2; // north
    } else if (neigh_idx == node_idx + _nx) {
        ns = 3; // south
    }
    return ns;
}

std::pair<size_t, size_t> Geometry::get_ij(size_t idx) const {
    size_t i = idx % _nx;
    size_t j = (idx / _nx) % _ny;
    return std::make_pair(i, j);
}

std::pair<size_t, size_t> Geometry::get_donor_ij(size_t ns) const {
    size_t from_i, from_j, to_i, to_j;
    std::tie(from_i, from_j) = get_ij(surfaces[ns].from_node);
    std::tie(to_i, to_j) = get_ij(surfaces[ns].to_node);

    // Determine donor subchannel based on flow direction
    size_t i_donor, j_donor;
    if (surfaces[ns].G >= 0.0) {
        i_donor = from_i; j_donor = from_j;
    } else {
        i_donor = to_i; j_donor = to_j;
    }

    return std::make_pair(i_donor, j_donor);
}

std::pair<size_t, size_t> Geometry::get_neighbor_ij(size_t i, size_t j, size_t ns) const {
    size_t i_neigh = i;
    size_t j_neigh = j;
    if (ns == 0) {

        // west neighbor
        if (i == 0) {
            i_neigh = -1;  // left boundary
        } else {
            i_neigh = i - 1;
        }

    } else if (ns == 1) {

        // east neighbor
        if (i == _nx - 1) {
            i_neigh = -1;  // right boundary
        } else {
            i_neigh = i + 1;
        }

    } else if (ns == 2) {

        // north neighbor
        if (j == 0) {
            j_neigh = -1;  // top boundary
        } else {
            j_neigh = j - 1;
        }

    } else if (ns == 3) {

        // south neighbor
        if (j == _ny - 1) {
            j_neigh = -1;  // bottom boundary
        } else {
            j_neigh = j + 1;
        }
    }

    return std::make_pair(i_neigh, j_neigh);
}
