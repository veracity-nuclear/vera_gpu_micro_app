#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <cstddef>
#include <utility>
#include <limits>
#include <vector>
#include <cmath>
#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>

#include "argument_parser.hpp"
#include "hdf5_kokkos.hpp"

// Forward declaration
class ArgumentParser;


struct Surface {
    size_t idx;         // surface index
    size_t from_node;   // upstream node index
    size_t to_node;     // downstream node index

    // Default constructor (required by Kokkos, must be trivial)
    KOKKOS_FUNCTION Surface() = default;

    // Parameterized constructor
    KOKKOS_FUNCTION Surface(size_t ns, size_t from, size_t to) : idx(ns), from_node(from), to_node(to) {}
};


template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class Geometry {

    using MemorySpace = typename ExecutionSpace::memory_space;
    using SurfacesView = Kokkos::View<Surface *, MemorySpace>;
    using View1D = Kokkos::View<double *, MemorySpace>;
    using View2D = Kokkos::View<double **, MemorySpace>;
    using View3D = Kokkos::View<double ***, MemorySpace>;
    using View4D = Kokkos::View<double ****, MemorySpace>;
    using ViewSizeT2D = Kokkos::View<size_t **, MemorySpace>;
    using ViewSizeT4D = Kokkos::View<size_t ****, MemorySpace>;

public:
    // Constructor for single assembly geometry
    Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, double length, size_t nchan, size_t naxial);
    // Constructor for full core geometry
    Geometry(const ArgumentParser& args);
    // Destructor
    ~Geometry() = default;

    SurfacesView surfaces;      // list of surfaces between subchannels

    const size_t boundary = std::numeric_limits<size_t>::max(); // for a neighbor that is a boundary
    double height() const { return H; }
    double flow_area(size_t aij, size_t k) const; // variable flow area
    double flow_area() const { return Af; } // average flow area for single assembly constructor
    double hydraulic_diameter() const { return Dh; }
    double gap_width() const { return gap_W; }
    double heated_perimeter() const { return 4.0 * Af / Dh; }
    double aspect_ratio() const { return gap_W / l; }
    double dz(size_t k) const; // variable axial spacing
    double dz() const { return H / _nz; } // average dz for single assembly constructor
    size_t nchan() const { return _nchan; }
    size_t core_size() const { return _core_map.extent(0); }
    size_t core_map(size_t aj, size_t ai) const { return _core_map(aj, ai); }
    size_t naxial() const { return _nz; }
    size_t nsurfaces() const;
    size_t nchannels() const { return _nchan * _nchan * nassemblies(); }
    size_t nassemblies() const;
    size_t global_chan_index(size_t aj, size_t ai, size_t j, size_t i) const { return _ij_global(aj, ai, j, i); }
    size_t global_surf_index(size_t aij, size_t ns) const;

private:
    double H;               // height of the subchannel
    double Af;              // flow area
    double Dh;              // hydraulic diameter
    double gap_W;           // gap width between subchannels
    double l;               // length of axial momentum cell
    size_t _nz;             // number of subchannels in x, y directions, and number of axial cells
    size_t _nchan;          // number of channels in x, y directions of an assembly (nchan x nchan)
    View2D _core_map;       // core map of assembly indices
    View4D _ij_global;      // mapping from (aj, ai, j, i) to global channel index
    ViewSizeT2D _ns_global;      // mapping from (aij, ns) to global surface index
    View1D _axial_mesh;     // axial mesh positions [m] (size: _nz+1)
    View2D _channel_area;   // channel flow areas [m^2] (size: nchannels x _nz)
};

