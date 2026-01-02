#include "geometry.hpp"

template <typename ExecutionSpace>
Geometry<ExecutionSpace>::Geometry(double height, double flow_area, double hydraulic_diameter, double gap_width, double length, size_t nchan, size_t naxial)
    : gap_W(gap_width), l(length), _nchan(nchan), _nz(naxial) {

    // Initialize core_map for single assembly (1x1 core)
    _core_size = 1;
    _core_sym = 1;
    _core_map = ViewSizeT2D("core_map", 1, 1);
    _core_map(0, 0) = 1;  // Single assembly

    // Initialize uniform axial mesh
    _axial_mesh = View1D("axial_mesh", _nz + 1);
    for (size_t k = 0; k <= _nz; ++k) {
        _axial_mesh(k) = k * (height / _nz);
    }

    // Initialize constant flow area for all channels
    size_t total_channels = _nchan * _nchan;
    _channel_area = View2D("channel_area", total_channels, _nz + 1);
    _hydraulic_diameter = View2D("hydraulic_diameter", total_channels, _nz + 1);
    for (size_t aij = 0; aij < total_channels; ++aij) {
        for (size_t k = 0; k < _nz + 1; ++k) {
            _channel_area(aij, k) = flow_area;
            _hydraulic_diameter(aij, k) = hydraulic_diameter;
        }
    }

    // Initialize global channel index mapping for single assembly
    _ij_global = ViewSizeT4D("ij_global", 1, 1, _nchan, _nchan);
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

    // output geometry info
    std::cout << "Single Assembly Geometry Initialized:" << std::endl;
    std::cout << "  Number of Channels (nchan x nchan): " << _nchan << " x " << _nchan << std::endl;
    std::cout << "  Number of Axial Cells (naxial): " << _nz << std::endl;
    std::cout << "  Total Assemblies: " << nassemblies() << std::endl;
    std::cout << "  Total Channels: " << nchannels() << std::endl;
    std::cout << "  Total Surfaces: " << nsurfaces() << std::endl;
}

template <typename ExecutionSpace>
Geometry<ExecutionSpace>::Geometry(const ArgumentParser& args) {

    // Extract parameters from ArgumentParser and initialize Geometry
    std::string filename = args.get_positional(0);

    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group core = file.getGroup("CORE");

    auto axial_mesh = HDF5ToKokkosView<View1D>(core.getDataSet("axial_mesh"), "axial_mesh"); // cm
    auto cell_height = HDF5ToKokkosView<View1D>(core.getDataSet("channel_cell_height"), "cell_height"); // cm
    auto pin_area = HDF5ToKokkosView<View4D>(core.getDataSet("pin_surface_area"), "pin_surface_area"); // cm^2
    auto channel_area = HDF5ToKokkosView<View4D>(core.getDataSet("channel_area"), "channel_area"); // cm^2

    auto _full_core_map = HDF5ToKokkosView<ViewSizeT2D>(core.getDataSet("core_map"), "core_map");
    _core_sym = core.getDataSet("core_sym").read<size_t>();
    _core_size = _full_core_map.extent(0);

    // extract SE quarter of core_map for quarter core symmetry
    if (_core_sym == 4) {
        size_t half_core_size = _core_size / 2;
        size_t new_core_size = (_core_size % 2 == 0) ? half_core_size : half_core_size + 1;
        _core_map = ViewSizeT2D("core_map", new_core_size, new_core_size);
        for (size_t aj = 0; aj < new_core_size; ++aj) {
            for (size_t ai = 0; ai < new_core_size; ++ai) {
                _core_map(aj, ai) = _full_core_map(half_core_size + aj, half_core_size + ai);
            }
        }
    } else {
        _core_map = _full_core_map;
    }

    std::cout << "\nFull Core Map" << std::endl;
    for (size_t aj = 0; aj < _full_core_map.extent(0); ++aj) {
        for (size_t ai = 0; ai < _full_core_map.extent(1); ++ai) {
            std::cout << std::setw(4) << (_full_core_map(aj, ai) == 0 ? "" : std::to_string(_full_core_map(aj, ai)));
        }
        std::cout << std::endl;
    }

    if (_core_sym == 4) {
        std::cout << "\nQuarter Core Map" << std::endl;
        for (size_t aj = 0; aj < _core_map.extent(0); ++aj) {
            for (size_t ai = 0; ai < _core_map.extent(1); ++ai) {
                std::cout << std::setw(4) << (_core_map(aj, ai) == 0 ? "" : std::to_string(_core_map(aj, ai)));
            }
            std::cout << std::endl;
        }
    }

    _core_size = _core_map.extent(0);
    _nz = axial_mesh.extent(0) - 1;
    _nchan = channel_area.extent(0);

    // Store axial mesh for variable spacing
    // convert axial mesh from cm to m
    for (size_t k = 0; k < axial_mesh.extent(0); ++k) {
        axial_mesh(k) *= 1e-2;
    }
    for (size_t k = 0; k < cell_height.extent(0); ++k) {
        cell_height(k) *= 1e-2;
    }
    _axial_mesh = axial_mesh;

    double apitch = core.getDataSet("apitch").read<double>() * 1e-2; // convert from cm to m
    double ppitch = apitch / _nchan; // approximation, pin pitch is not in VERAout CORE group
    double length = ppitch; // approximation for channel cells inbetween assemblies
    double gap_width = ppitch * 0.5; // approximation, gap width is not in VERAout CORE group

    gap_W = gap_width;
    l = length;

    // allocate global channel index mapping View4D with shape (core_size, core_size, nchan, nchan)
    _ij_global = ViewSizeT4D("ij_global", _core_size, _core_size, _nchan, _nchan);

    // fill global channel index mapping
    size_t global_idx = 0;
    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t ai = 0; ai < _core_size; ++ai) {
            if (_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    _ij_global(aj, ai, j, i) = global_idx;
                    global_idx++;
                }
            }
        }
    }

    std::cout << "\nTotal assemblies in core: " << nassemblies() << std::endl;
    std::cout << "Total channels in core: " << nchannels() << std::endl;

    // Flatten channel_area from 4D (nchan, nchan, nz, nassembly) to 2D (nchannels, nz)
    _channel_area = View2D("channel_area", nchannels(), _nz);
    _hydraulic_diameter = View2D("hydraulic_diameter", nchannels(), _nz);

    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t ai = 0; ai < _core_size; ++ai) {
            if (_core_map(aj, ai) == 0) continue;
            size_t assem_idx = _core_map(aj, ai) - 1; // Convert to 0-based index
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    size_t aij = _ij_global(aj, ai, j, i);
                    for (size_t k = 0; k < _nz; ++k) {

                        _channel_area(aij, k) = channel_area(i, j, k, assem_idx) * 1e-4; // convert from cm^2 to m^2

                        // Check 4 neighboring pins (SW, SE, NW, NE) to calculate the wetted perimeter
                        // Subchannel (j,i) is bounded by pins:
                        // SW: (j, i-1), SE: (j, i), NW: (j-1, i-1), NE: (j-1, i)

                        double A_wetted = 0.0; // convert from cm^2 to m^2
                        if (j > 0 && i > 0) { // NW pin exists
                            A_wetted += 0.25 * pin_area(i - 1, j - 1, k, assem_idx) * 1e-4;
                        }
                        if (j > 0 && i < npin()) { // NE pin exists
                            A_wetted += 0.25 * pin_area(i, j - 1, k, assem_idx) * 1e-4;
                        }
                        if (j < npin() && i > 0) { // SW pin exists
                            A_wetted += 0.25 * pin_area(i - 1, j, k, assem_idx) * 1e-4;
                        }
                        if (j < npin() && i < npin()) { // SE pin exists
                            A_wetted += 0.25 * pin_area(i, j, k, assem_idx) * 1e-4;
                        }
                        double P_wetted = A_wetted / cell_height(k);
                        _hydraulic_diameter(aij, k) = 4.0 * _channel_area(aij, k) / P_wetted;
                    }
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
    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t j = 0; j < _nchan; ++j) {
            for (size_t ai = 0; ai < _core_size; ++ai) {
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
                    else if (ai + 1 < _core_size && _core_map(aj, ai + 1) > 0) {
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
    for (size_t ai = 0; ai < _core_size; ++ai) {
        for (size_t i = 0; i < _nchan; ++i) {
            for (size_t aj = 0; aj < _core_size; ++aj) {
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
                    else if (aj + 1 < _core_size && _core_map(aj + 1, ai) > 0) {
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

    std::cout << "\n=== GEOMETRY SUMMARY ===" << std::endl;
    std::cout << "Assembly size: (" << _nchan << ", " << _nchan << ")" << std::endl;
    std::cout << "Core size: (" << _full_core_map.extent(0) << ", " << _full_core_map.extent(1) << ")" << std::endl;
    std::cout << "Core symmetry: " << _core_sym << std::endl;
    std::cout << "Nsurfaces: " << nsurfaces() << std::endl;
    std::cout << "Nchannels: " << nchannels() << std::endl;
    std::cout << "Nassemblies: " << nassemblies() << std::endl;
    std::cout << "Naxial: " << _nz << std::endl;
    std::cout << "Height: " << core_height() << " m" << std::endl;

    // Compute and print axial mesh statistics
    double dz_min = std::numeric_limits<double>::max();
    double dz_max = std::numeric_limits<double>::lowest();
    double dz_sum = 0.0;
    for (size_t k = 0; k < _nz; ++k) {
        double dz_k = _axial_mesh(k + 1) - _axial_mesh(k);
        dz_min = std::min(dz_min, dz_k);
        dz_max = std::max(dz_max, dz_k);
        dz_sum += dz_k;
    }
    double dz_avg = dz_sum / _nz;
    std::cout << "\nAxial Mesh Spacing [m]:" << std::endl;
    std::cout << "  Min: " << dz_min << ", Max: " << dz_max << ", Avg: " << dz_avg << std::endl;

    // Compute and print channel area statistics
    double area_min = std::numeric_limits<double>::max();
    double area_max = std::numeric_limits<double>::lowest();
    double area_sum = 0.0;
    size_t area_count = 0;
    for (size_t aij = 0; aij < nchannels(); ++aij) {
        for (size_t k = 0; k < _nz; ++k) {
            double val = _channel_area(aij, k);
            if (val > 1e-12) { // Skip zero-area channels
                area_min = std::min(area_min, val);
                area_max = std::max(area_max, val);
                area_sum += val;
                area_count++;
            }
        }
    }
    double area_avg = area_sum / area_count;
    std::cout << "\nChannel Flow Area [m²]:" << std::endl;
    std::cout << "  Min: " << area_min << ", Max: " << area_max << ", Avg: " << area_avg << std::endl;

    // Compute and print hydraulic diameter statistics
    double Dh_min = std::numeric_limits<double>::max();
    double Dh_max = std::numeric_limits<double>::lowest();
    double Dh_sum = 0.0;
    size_t Dh_count = 0;
    for (size_t aij = 0; aij < nchannels(); ++aij) {
        for (size_t k = 0; k < _nz; ++k) {
            double val = _hydraulic_diameter(aij, k);
            if (val > 1e-12) { // Skip zero-area channels
                Dh_min = std::min(Dh_min, val);
                Dh_max = std::max(Dh_max, val);
                Dh_sum += val;
                Dh_count++;
            }
        }
    }
    double Dh_avg = Dh_sum / Dh_count;
    std::cout << "\nHydraulic Diameter [m]:" << std::endl;
    std::cout << "  Min: " << Dh_min << ", Max: " << Dh_max << ", Avg: " << Dh_avg << std::endl;

    std::cout << "\nGap Width: " << gap_W << " m" << std::endl;
    std::cout << "Length: " << l << " m" << std::endl;

    std::cout << "========================\n" << std::endl;
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

// Explicit template instantiations
template class Geometry<Kokkos::DefaultExecutionSpace>;
template class Geometry<Kokkos::Serial>;
#if defined(KOKKOS_ENABLE_SERIAL) && !defined(KOKKOS_ENABLE_OPENMP)
template class Geometry<Kokkos::Serial>;
#endif
