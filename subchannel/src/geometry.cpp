#include "geometry.hpp"

template <typename ExecutionSpace>
Geometry<ExecutionSpace>::Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, double length, size_t nchan, size_t naxial)
    : H(height), Af(flow_area), Dh(hydraulic_diameter), gap_W(gap_width), l(length), _nchan(nchan), _nz(naxial) {

    // Initialize core_map for single assembly (1x1 core)
    size_t core_size = 1;
    _core_map = View2D("core_map", 1, 1);
    _core_map(0, 0) = 1;  // Single assembly

    // Initialize global channel index mapping for single assembly
    _ij_global = View4D("ij_global", 1, 1, _nchan, _nchan);
    size_t global_idx = 0;
    for (size_t j = 0; j < _nchan; ++j) {
        for (size_t i = 0; i < _nchan; ++i) {
            _ij_global(0, 0, j, i) = global_idx++;
        }
    }

    // Allocate the surfaces View
    surfaces = SurfacesView("surfaces", nsurfaces());

    // Allocate _ns_global for mapping channels to surfaces
    _ns_global = ViewSizeT2D("ns_global", nchannels(), 4);

    // Initialize to boundary
    for (size_t aij = 0; aij < nchannels(); ++aij) {
        for (size_t ns = 0; ns < 4; ++ns) {
            _ns_global(aij, ns) = boundary;
        }
    }

    // create all W -> E surfaces in between subchannels for any axial plane
    size_t ns = 0; // surface index
    for (size_t j = 0; j < _nchan; ++j) {
        for (size_t i = 0; i < _nchan; ++i) {

            size_t aij = _ij_global(0, 0, j, i);  // aij the flattened global channel index

            // west -> east surfaces in assembly at (aj, ai)
            if (i + 1 < _nchan) {
                size_t aij_neigh = _ij_global(0, 0, j, i + 1);
                surfaces(ns) = Surface(ns, aij, aij_neigh);
                _ns_global(aij_neigh, 0) = ns;  // ns is the neighbor channel's west surface
                _ns_global(aij, 1) = ns;        // ns is the current channel's east surface
                ns++;
            }
        }
    }

    // create all N -> S surfaces in between subchannels for any axial plane
    for (size_t i = 0; i < _nchan; ++i) {
        for (size_t j = 0; j < _nchan; ++j) {

            size_t aij = _ij_global(0, 0, j, i);  // aij the flattened global channel index

            // north -> south surfaces in assembly at (aj, ai)
            if (j + 1 < _nchan) {
                size_t aij_neigh = _ij_global(0, 0, j + 1, i);
                surfaces(ns) = Surface(ns, aij, aij_neigh);
                _ns_global(aij_neigh, 2) = ns;  // ns is the neighbor channel's north surface
                _ns_global(aij, 3) = ns;        // ns is the current channel's south surface
                ns++;
            }
        }
    }
}

template <typename ExecutionSpace>
Geometry<ExecutionSpace>::Geometry(const ArgumentParser& args) {

    // Extract parameters from ArgumentParser and initialize Geometry
    std::string filename = args.get_positional(0);

    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group core = file.getGroup("CORE");

    auto axial_mesh = HDF5ToKokkosView<View1D>(core.getDataSet("axial_mesh"), "axial_mesh");
    auto cell_height = HDF5ToKokkosView<View1D>(core.getDataSet("channel_cell_height"), "cell_height");
    auto pin_area = HDF5ToKokkosView<View4D>(core.getDataSet("pin_surface_area"), "pin_surface_area");
    auto channel_area = HDF5ToKokkosView<View4D>(core.getDataSet("channel_area"), "channel_area");

    _core_map = HDF5ToKokkosView<View2D>(core.getDataSet("core_map"), "core_map");
    _nz = axial_mesh.extent(0) - 1;
    _nchan = channel_area.extent(0);

    double lhr = core.getDataSet("nominal_linear_heat_rate").read<double>(); // will improve with Issue #95
    double height = axial_mesh(axial_mesh.extent(0) - 1) - axial_mesh(0);
    double flow_area = channel_area(_nchan / 2, _nchan / 2, _nz / 2, 0); // pick center assembly at midplane (will improve with Issue #94)
    double pin_area_avg = pin_area(_nchan / 2, _nchan / 2, _nz / 2, 0); // pick center assembly at midplane
    double pin_diameter_avg = pin_area_avg / (M_PI * cell_height(_nz / 2));
    double pin_circumference_avg = M_PI * pin_diameter_avg;
    double hydraulic_diameter = 4.0 * flow_area / pin_circumference_avg; // approximation with the avg pin circumference
    double apitch = core.getDataSet("apitch").read<double>();
    double ppitch = apitch / _nchan; // approximation, pin pitch is not in VERAout CORE group
    double length = ppitch; // approximation for channel cells inbetween assemblies
    double gap_width = ppitch - pin_diameter_avg;

    H = height;
    Af = flow_area;
    Dh = hydraulic_diameter;
    gap_W = gap_width;
    l = length;

    // allocate global channel index mapping View4D with shape (core_size, core_size, nchan, nchan)
    size_t core_size = _core_map.extent(0);
    _ij_global = View4D("ij_global", core_size, core_size, _nchan, _nchan);

    // fill global channel index mapping
    size_t global_idx = 0;
    for (size_t aj = 0; aj < core_size; ++aj) {
        for (size_t ai = 0; ai < core_size; ++ai) {
            if (_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    _ij_global(aj, ai, j, i) = global_idx;
                    global_idx++;
                }
            }
        }
    }

    // allocate memory for surfaces and channels Views
    surfaces = SurfacesView("surfaces", nsurfaces());
    _ns_global = ViewSizeT2D("ns_global", nchannels(), 4); // 4 surfaces per channel: W, E, N, S

    // initialize to boundary
    for (size_t aij = 0; aij < nchannels(); ++aij) {
        for (size_t ns = 0; ns < 4; ++ns) {
            _ns_global(aij, ns) = boundary;
        }
    }

    // create all W -> E surfaces in between subchannels for any axial plane
    size_t ns = 0; // surface index
    for (size_t aj = 0; aj < core_size; ++aj) {
        for (size_t j = 0; j < _nchan; ++j) {
            for (size_t ai = 0; ai < core_size; ++ai) {
                if (_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
                for (size_t i = 0; i < _nchan; ++i) {

                    size_t aij = _ij_global(aj, ai, j, i);  // aij the flattened global channel index

                    // west -> east surfaces in assembly at (aj, ai)
                    if (i + 1 < _nchan) {
                        size_t aij_neigh = _ij_global(aj, ai, j, i + 1);
                        surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _ns_global(aij_neigh, 0) = ns;  // ns is the neighbor channel's west surface
                        _ns_global(aij, 1) = ns;        // ns is the current channel's east surface
                        ns++;
                    }

                    // east assembly neighbor surface
                    else if (ai + 1 < core_size && _core_map(aj, ai + 1) > 0) {
                        size_t aij_neigh = _ij_global(aj, ai + 1, j, 0);
                        surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _ns_global(aij_neigh, 0) = ns;  // ns is the neighbor channel's west surface
                        _ns_global(aij, 1) = ns;        // ns is the current channel's east surface
                        ns++;
                    }
                }
            }
        }
    }

    // create all N -> S surfaces in between subchannels for any axial plane
    for (size_t ai = 0; ai < core_size; ++ai) {
        for (size_t i = 0; i < _nchan; ++i) {
            for (size_t aj = 0; aj < core_size; ++aj) {
                if (_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
                for (size_t j = 0; j < _nchan; ++j) {

                    size_t aij = _ij_global(aj, ai, j, i);  // aij the flattened global channel index

                    // north -> south surfaces in assembly at (aj, ai)
                    if (j + 1 < _nchan) {
                        size_t aij_neigh = _ij_global(aj, ai, j + 1, i);
                        surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _ns_global(aij_neigh, 2) = ns;  // ns is the neighbor channel's north surface
                        _ns_global(aij, 3) = ns;        // ns is the current channel's south surface
                        ns++;
                    }

                    // south assembly neighbor surface
                    else if (aj + 1 < core_size && _core_map(aj + 1, ai) > 0) {
                        size_t aij_neigh = _ij_global(aj + 1, ai, 0, i);
                        surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _ns_global(aij_neigh, 2) = ns;  // ns is the neighbor channel's north surface
                        _ns_global(aij, 3) = ns;        // ns is the current channel's south surface
                        ns++;
                    }
                }
            }
        }
    }

    std::cout << "Assembly size: (" << _nchan << ", " << _nchan << ")" << std::endl;
    std::cout << "Core size: (" << _core_map.extent(0) << ", " << _core_map.extent(1) << ")" << std::endl;
    std::cout << "Nsurfaces: " << nsurfaces() << std::endl;
    std::cout << "Nchannels: " << nchannels() << std::endl;
    std::cout << "Nassemblies: " << nassemblies() << std::endl;
    std::cout << "Naxial: " << _nz << std::endl;
    std::cout << "LHR: " << lhr << " W/m" << std::endl;
    std::cout << "Height: " << height << " cm" << std::endl;
    std::cout << "Flow area: " << flow_area << " cm^2" << std::endl;
    std::cout << "Hydraulic diameter: " << hydraulic_diameter << " cm" << std::endl;
    std::cout << "Gap width: " << gap_width << " cm" << std::endl;
    std::cout << "Length: " << length << " cm" << std::endl;
    std::cout << "Assembly pitch: " << apitch << " cm" << std::endl;

    std::cout << "Core Map:" << std::endl;
    for (size_t i = 0; i < _core_map.extent(0); ++i) {
        for (size_t j = 0; j < _core_map.extent(1); ++j) {
            if (_core_map(i, j) == 0) std::cout << std::setw(5) << ' ';
            else std::cout << std::setw(5) << _core_map(i, j);
        }
        std::cout << '\n' << std::endl;
    }

}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::nsurfaces() const {
    // Total number of internal surfaces in the subchannel grid
    size_t nsurf = 0;
    for (size_t aj = 0; aj < _core_map.extent(0); ++aj) {
        for (size_t ai = 0; ai < _core_map.extent(1); ++ai) {
            if (_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
            nsurf += (_nchan - 1) * _nchan * 2; // vertical and horizontal surfaces
            if (ai + 1 < core_size() && _core_map(aj, ai + 1) > 0) nsurf += _nchan; // East assembly neighbor surfaces
            if (aj + 1 < core_size() && _core_map(aj + 1, ai) > 0) nsurf += _nchan; // South assembly neighbor surfaces
        }
    }

    return nsurf;
}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::nassemblies() const {
    // Total number of fuel assemblies in the core
    size_t nassy = 0;
    for (size_t i = 0; i < _core_map.extent(0); ++i) {
        for (size_t j = 0; j < _core_map.extent(1); ++j) {
            if (_core_map(i, j) > 0) ++nassy;
        }
    }
    return nassy;
}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::global_surf_index(size_t aij, size_t ns) const {
    // Calculate global surface index based on subchannel index (aij) and surface number (ns)
    // Surface numbering: 0 = west, 1 = east, 2 = north, 3 = south
    return _ns_global(aij, ns);
}

// Explicit template instantiations
template class Geometry<Kokkos::DefaultExecutionSpace>;
template class Geometry<Kokkos::Serial>;
#if defined(KOKKOS_ENABLE_SERIAL) && !defined(KOKKOS_ENABLE_OPENMP)
template class Geometry<Kokkos::Serial>;
#endif
