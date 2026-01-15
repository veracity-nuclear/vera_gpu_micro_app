#include "geometry.hpp"
#include <numeric>

template <typename ExecutionSpace>
Geometry<ExecutionSpace>::Geometry(
    double height,
    double flow_area,
    double hydraulic_diameter,
    double gap_width,
    double length,
    size_t nchan,
    size_t naxial,
    ViewSizeT2D core_map
)
    : gap_W(gap_width), l(length), _nchan(nchan), _nz(naxial) {

    // Initialize core_map
    _core_map = core_map;
    _core_size = _core_map.extent(0);
    _core_sym = 1;

    auto _h_core_map = Kokkos::create_mirror_view(_core_map);
    Kokkos::deep_copy(_h_core_map, _core_map);

    // Initialize uniform axial mesh
    _axial_mesh = View1D("axial_mesh", _nz + 1);
    auto _h_axial_mesh = Kokkos::create_mirror_view(_axial_mesh);
    Kokkos::deep_copy(_h_axial_mesh, _axial_mesh);
    for (size_t k = 0; k <= _nz; ++k) {
        _h_axial_mesh(k) = k * (height / _nz);
    }
    Kokkos::deep_copy(_axial_mesh, _h_axial_mesh);

    // Initialize constant flow area for all channels
    _channel_area = View2D("channel_area", nchannels(), _nz + 1);
    _hydraulic_diameter = View2D("hydraulic_diameter", nchannels(), _nz + 1);
    auto _h_channel_area = Kokkos::create_mirror_view(_channel_area);
    auto _h_hydraulic_diameter = Kokkos::create_mirror_view(_hydraulic_diameter);
    Kokkos::deep_copy(_h_channel_area, _channel_area);
    Kokkos::deep_copy(_h_hydraulic_diameter, _hydraulic_diameter);
    for (size_t aij = 0; aij < nchannels(); ++aij) {
        for (size_t k = 0; k < _nz + 1; ++k) {
            _h_channel_area(aij, k) = flow_area;
            _h_hydraulic_diameter(aij, k) = hydraulic_diameter;
        }
    }
    Kokkos::deep_copy(_channel_area, _h_channel_area);
    Kokkos::deep_copy(_hydraulic_diameter, _h_hydraulic_diameter);

    // Initialize global channel index mapping
    _ij_global = ViewSizeT4D("ij_global", _core_size, _core_size, _nchan, _nchan);
    auto _h_ij_global = Kokkos::create_mirror_view(_ij_global);
    Kokkos::deep_copy(_h_ij_global, _ij_global);
    size_t global_idx = 0;
    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t ai = 0; ai < _core_size; ++ai) {
            if (_h_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    _h_ij_global(aj, ai, j, i) = global_idx++;
                }
            }
        }
    }
    Kokkos::deep_copy(_ij_global, _h_ij_global);

    // Allocate the surfaces View
    surfaces = SurfacesView("surfaces", nsurfaces());
    auto _h_surfaces = Kokkos::create_mirror_view(surfaces);
    Kokkos::deep_copy(_h_surfaces, surfaces);

    // Allocate _ns_global for mapping channels to surfaces
    _ns_global = ViewSizeT2D("ns_global", nchannels(), 4);
    auto _h_ns_global = Kokkos::create_mirror_view(_ns_global);
    Kokkos::deep_copy(_h_ns_global, _ns_global);

    // Initialize to boundary
    for (size_t aij = 0; aij < nchannels(); ++aij) {
        for (size_t ns = 0; ns < 4; ++ns) {
            _h_ns_global(aij, ns) = boundary;
        }
    }

    // create all W -> E surfaces in between subchannels for any axial plane
    size_t ns = 0; // surface index
    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t j = 0; j < _nchan; ++j) {
            for (size_t ai = 0; ai < _core_size; ++ai) {
                if (_h_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
                for (size_t i = 0; i < _nchan; ++i) {

                    size_t aij = _h_ij_global(aj, ai, j, i);  // aij the flattened global channel index

                    // west -> east surfaces in assembly at (aj, ai)
                    if (i + 1 < _nchan) {
                        size_t aij_neigh = _h_ij_global(aj, ai, j, i + 1);
                        _h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 0) = ns;  // ns is the neighbor channel's west surface
                        _h_ns_global(aij, 1) = ns;        // ns is the current channel's east surface
                        ns++;
                    }

                    // east assembly neighbor surface
                    else if (ai + 1 < _core_size && _h_core_map(aj, ai + 1) > 0) {
                        size_t aij_neigh = _h_ij_global(aj, ai + 1, j, 0);
                        _h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 0) = ns;  // ns is the neighbor channel's west surface
                        _h_ns_global(aij, 1) = ns;        // ns is the current channel's east surface
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
                if (_h_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
                for (size_t j = 0; j < _nchan; ++j) {

                    size_t aij = _h_ij_global(aj, ai, j, i);  // aij the flattened global channel index

                    // north -> south surfaces in assembly at (aj, ai)
                    if (j + 1 < _nchan) {
                        size_t aij_neigh = _h_ij_global(aj, ai, j + 1, i);
                        _h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 2) = ns;  // ns is the neighbor channel's north surface
                        _h_ns_global(aij, 3) = ns;        // ns is the current channel's south surface
                        ns++;
                    }

                    // south assembly neighbor surface
                    else if (aj + 1 < _core_size && _h_core_map(aj + 1, ai) > 0) {
                        size_t aij_neigh = _h_ij_global(aj + 1, ai, 0, i);
                        _h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 2) = ns;  // ns is the neighbor channel's north surface
                        _h_ns_global(aij, 3) = ns;        // ns is the current channel's south surface
                        ns++;
                    }
                }
            }
        }
    }
    Kokkos::deep_copy(surfaces, _h_surfaces);
    Kokkos::deep_copy(_ns_global, _h_ns_global);

    // output geometry info
    std::cout << "Geometry Initialized:" << std::endl;
    std::cout << "  Number of Channels per Assembly (nchan x nchan): " << _nchan << " x " << _nchan << std::endl;
    std::cout << "  Core Size: " << _core_size << " x " << _core_size << std::endl;
    std::cout << "  Number of Axial Cells (naxial): " << _nz << std::endl;
    std::cout << "  Total Assemblies: " << nassemblies() << std::endl;
    std::cout << "  Total Channels: " << nchannels() << std::endl;
    std::cout << "  Total Surfaces: " << nsurfaces() << std::endl;

    // Build surface connectivity
    build_surface_connectivity();
}

template <typename ExecutionSpace>
Geometry<ExecutionSpace>::Geometry(const ArgumentParser& args) {

    // Extract parameters from ArgumentParser and initialize Geometry
    std::string filename = args.get_positional(0);

    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group core = file.getGroup("CORE");

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

    auto axial_mesh = HDF5ToKokkosView<View1D>(core.getDataSet("axial_mesh"), "axial_mesh"); // cm
    _nz = axial_mesh.extent(0) - 1;
    for (size_t k = 0; k < axial_mesh.extent(0); ++k) {
        axial_mesh(k) *= 1e-2;
    }
    _axial_mesh = axial_mesh;

    auto pin_volumes = HDF5ToKokkosView<View4D>(core.getDataSet("pin_volumes"), "pin_volumes"); // cm^3
    _nchan = pin_volumes.extent(0) + 1;

    std::cout << "\n=== GEOMETRY SUMMARY ===" << std::endl;
    std::cout << "Assembly size: (" << _nchan << ", " << _nchan << ")" << std::endl;

    double apitch = core.getDataSet("apitch").read<double>() * 1e-2; // convert from cm to m
    double ppitch = apitch / _nchan; // approximation, pin pitch is not in VERAout CORE group
    double length = ppitch; // approximation for channel cells inbetween assemblies
    double gap_width = ppitch * 0.5; // approximation, gap width is not in VERAout CORE group

    View4D channel_area; // cm^2
    auto h_channel_area = Kokkos::create_mirror_view(channel_area);
    // if (core.exist("channel_area")) {
    //     channel_area = HDF5ToKokkosView<View4D>(core.getDataSet("channel_area"), "channel_area"); // cm^2
    // } else {
    //     double default_area_cm2 = ppitch * ppitch * 1e4; // cm^2
    //     h_channel_area = _init_default_channel_area(default_area_cm2);
    //     Kokkos::deep_copy(channel_area, h_channel_area);
    // }

    View4D pin_area; // cm^2
    if (core.exist("pin_surface_area")) {
        pin_area = HDF5ToKokkosView<View4D>(core.getDataSet("pin_surface_area"), "pin_surface_area"); // cm^2
    } else {
        pin_area = _init_default_pin_area(pin_volumes);
    }

    gap_W = gap_width;
    l = length;

    // allocate global channel index mapping View4D with shape (core_size, core_size, nchan, nchan)
    _ij_global = ViewSizeT4D("ij_global", _core_size, _core_size, _nchan, _nchan);
    auto _h_ij_global = Kokkos::create_mirror_view(_ij_global);

    // fill global channel index mapping
    size_t global_idx = 0;
    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t ai = 0; ai < _core_size; ++ai) {
            if (_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    _h_ij_global(aj, ai, j, i) = global_idx;
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
    auto _h_channel_area = Kokkos::create_mirror_view(_channel_area);
    auto _h_hydraulic_diameter = Kokkos::create_mirror_view(_hydraulic_diameter);

    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t ai = 0; ai < _core_size; ++ai) {
            if (_core_map(aj, ai) == 0) continue;
            size_t assem_idx = _core_map(aj, ai) - 1; // Convert to 0-based index
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    size_t aij = _ij_global(aj, ai, j, i);
                    for (size_t k = 0; k < _nz; ++k) {

                        _h_channel_area(aij, k) = channel_area(i, j, k, assem_idx) * 1e-4; // convert from cm^2 to m^2

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
                        double P_wetted = A_wetted / dz(k);
                        if (P_wetted == 0.0) {
                            _h_hydraulic_diameter(aij, k) = std::sqrt(4.0 * _h_channel_area(aij, k) / M_PI); // approximate with circular-equivalent hydraulic diameter
                        } else {
                            _h_hydraulic_diameter(aij, k) = 4.0 * _h_channel_area(aij, k) / P_wetted;
                        }
                    }
                }
            }
        }
    }
    Kokkos::deep_copy(_channel_area, _h_channel_area);
    Kokkos::deep_copy(_hydraulic_diameter, _h_hydraulic_diameter);

    // allocate memory for surfaces and channels Views
    surfaces = SurfacesView("surfaces", nsurfaces());
    _ns_global = ViewSizeT2D("ns_global", nchannels(), 4); // 4 surfaces per channel: W, E, N, S
    auto h_surfaces = Kokkos::create_mirror_view(surfaces);
    auto _h_ns_global = Kokkos::create_mirror_view(_ns_global);

    // initialize to boundary
    for (size_t aij = 0; aij < nchannels(); ++aij) {
        for (size_t ns = 0; ns < 4; ++ns) {
            _h_ns_global(aij, ns) = boundary;
        }
    }

    // create all W -> E surfaces in between subchannels for any axial plane
    size_t ns = 0; // surface index
    for (size_t aj = 0; aj < _core_size; ++aj) {
        for (size_t j = 0; j < _nchan; ++j) {
            for (size_t ai = 0; ai < _core_size; ++ai) {
                if (_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
                for (size_t i = 0; i < _nchan; ++i) {

                    size_t aij = _h_ij_global(aj, ai, j, i);  // aij the flattened global channel index

                    // west -> east surfaces in assembly at (aj, ai)
                    if (i + 1 < _nchan) {
                        size_t aij_neigh = _h_ij_global(aj, ai, j, i + 1);
                        h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 0) = ns;  // ns is the neighbor channel's west surface
                        _h_ns_global(aij, 1) = ns;        // ns is the current channel's east surface
                        ns++;
                    }

                    // east assembly neighbor surface
                    else if (ai + 1 < _core_size && _core_map(aj, ai + 1) > 0) {
                        size_t aij_neigh = _h_ij_global(aj, ai + 1, j, 0);
                        h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 0) = ns;  // ns is the neighbor channel's west surface
                        _h_ns_global(aij, 1) = ns;        // ns is the current channel's east surface
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

                    size_t aij = _h_ij_global(aj, ai, j, i);  // aij the flattened global channel index

                    // north -> south surfaces in assembly at (aj, ai)
                    if (j + 1 < _nchan) {
                        size_t aij_neigh = _h_ij_global(aj, ai, j + 1, i);
                        h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 2) = ns;  // ns is the neighbor channel's north surface
                        _h_ns_global(aij, 3) = ns;        // ns is the current channel's south surface
                        ns++;
                    }

                    // south assembly neighbor surface
                    else if (aj + 1 < _core_size && _core_map(aj + 1, ai) > 0) {
                        size_t aij_neigh = _h_ij_global(aj + 1, ai, 0, i);
                        h_surfaces(ns) = Surface(ns, aij, aij_neigh);
                        _h_ns_global(aij_neigh, 2) = ns;  // ns is the neighbor channel's north surface
                        _h_ns_global(aij, 3) = ns;        // ns is the current channel's south surface
                        ns++;
                    }
                }
            }
        }
    }

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

    // Build surface connectivity
    build_surface_connectivity();
}

template <typename ExecutionSpace>
double Geometry<ExecutionSpace>::core_height() const {
    auto _h_axial_mesh = Kokkos::create_mirror_view(_axial_mesh);
    Kokkos::deep_copy(_h_axial_mesh, _axial_mesh);
    return _h_axial_mesh(_nz - 1) - _h_axial_mesh(0);
}

template <typename ExecutionSpace>
double Geometry<ExecutionSpace>::flow_area(size_t aij, size_t k) const {
    auto _h_channel_area = Kokkos::create_mirror_view(_channel_area);
    Kokkos::deep_copy(_h_channel_area, _channel_area);
    return _h_channel_area(aij, k);
}

template <typename ExecutionSpace>
double Geometry<ExecutionSpace>::hydraulic_diameter(size_t aij, size_t k) const {
    auto _h_hydraulic_diameter = Kokkos::create_mirror_view(_hydraulic_diameter);
    Kokkos::deep_copy(_h_hydraulic_diameter, _hydraulic_diameter);
    return _h_hydraulic_diameter(aij, k);
}

template <typename ExecutionSpace>
double Geometry<ExecutionSpace>::dz(size_t k) const {
    auto _h_axial_mesh = Kokkos::create_mirror_view(_axial_mesh);
    Kokkos::deep_copy(_h_axial_mesh, _axial_mesh);
    return _h_axial_mesh(k + 1) - _h_axial_mesh(k);
}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::core_map(size_t aj, size_t ai) const {
    auto _h_core_map = Kokkos::create_mirror_view(_core_map);
    Kokkos::deep_copy(_h_core_map, _core_map);
    return _h_core_map(aj, ai);
}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::nsurfaces() const {

    auto _h_core_map = Kokkos::create_mirror_view(_core_map);
    Kokkos::deep_copy(_h_core_map, _core_map);

    // Total number of internal surfaces in the subchannel grid
    size_t nsurf = 0;
    for (size_t aj = 0; aj < _h_core_map.extent(0); ++aj) {
        for (size_t ai = 0; ai < _h_core_map.extent(1); ++ai) {
            if (_h_core_map(aj, ai) == 0) continue; // skip non-existent assemblies
            nsurf += (_nchan - 1) * _nchan * 2; // vertical and horizontal surfaces
            if (ai + 1 < core_size() && _h_core_map(aj, ai + 1) > 0) nsurf += _nchan; // East assembly neighbor surfaces
            if (aj + 1 < core_size() && _h_core_map(aj + 1, ai) > 0) nsurf += _nchan; // South assembly neighbor surfaces
        }
    }

    return nsurf;
}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::nassemblies() const {

    auto _h_core_map = Kokkos::create_mirror_view(_core_map);
    Kokkos::deep_copy(_h_core_map, _core_map);

    // Total number of fuel assemblies in the core
    size_t nassy = 0;
    for (size_t i = 0; i < _h_core_map.extent(0); ++i) {
        for (size_t j = 0; j < _h_core_map.extent(1); ++j) {
            if (_h_core_map(i, j) > 0) ++nassy;
        }
    }
    return nassy;
}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::global_chan_index(size_t aj, size_t ai, size_t j, size_t i) const {
    auto _h_ij_global = Kokkos::create_mirror_view(_ij_global);
    Kokkos::deep_copy(_h_ij_global, _ij_global);
    return _h_ij_global(aj, ai, j, i);
}

template <typename ExecutionSpace>
size_t Geometry<ExecutionSpace>::global_surf_index(size_t aij, size_t ns) const {
    auto _h_ns_global = Kokkos::create_mirror_view(_ns_global);
    Kokkos::deep_copy(_h_ns_global, _ns_global);
    return _h_ns_global(aij, ns);
}

template <typename ExecutionSpace>
typename Geometry<ExecutionSpace>::View4D Geometry<ExecutionSpace>::_init_default_channel_area(double default_area_cm2) {
    // Initialize default channel area View with uniform values
    View4D channel_area("channel_area", _nchan, _nchan, _nz, nassemblies());
    auto _h_channel_area = Kokkos::create_mirror_view(channel_area);
    for (size_t a = 0; a < nassemblies(); ++a) {
        for (size_t k = 0; k < _nz; ++k) {
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    _h_channel_area(i, j, k, a) = default_area_cm2; // cm^2
                }
            }
        }
    }
    Kokkos::deep_copy(channel_area, _h_channel_area);
    return channel_area;
}

template <typename ExecutionSpace>
typename Geometry<ExecutionSpace>::View4D Geometry<ExecutionSpace>::_init_default_pin_area(const View4D& pin_volumes) {

    auto _h_pin_volumes = Kokkos::create_mirror_view(pin_volumes);
    Kokkos::deep_copy(_h_pin_volumes, pin_volumes);

    // Initialize default channel area View with uniform values
    View4D pin_area("pin_area", _nchan, _nchan, _nz, nassemblies());
    auto _h_pin_area = Kokkos::create_mirror_view(pin_area);
    for (size_t a = 0; a < nassemblies(); ++a) {
        for (size_t k = 0; k < _nz; ++k) {
            for (size_t j = 0; j < _nchan; ++j) {
                for (size_t i = 0; i < _nchan; ++i) {
                    _h_pin_area(i, j, k, a) = _h_pin_volumes(i, j, k, a) / (dz(k) * 1e2); // cm^2 (dz returns [m], so convert to cm before dividing)
                }
            }
        }
    }
    Kokkos::deep_copy(pin_area, _h_pin_area);
    return pin_area;
}

template <typename ExecutionSpace>
void Geometry<ExecutionSpace>::build_surface_connectivity() {
    const size_t nsurf = nsurfaces();
    const size_t max_neighbors = max_surface_connectivity(); // max number of surfaces that a surface can non-trivially perturb

    // Allocate connectivity views
    _num_neighbors = ViewSizeT1D("num_neighbors", nsurf);
    _surface_neighbors = ViewSizeT2D("surface_neighbors", nsurf, max_neighbors);

    auto h_surface_neighbors = Kokkos::create_mirror_view(_surface_neighbors);
    auto h_num_neighbors = Kokkos::create_mirror_view(_num_neighbors);
    auto h_surfaces = Kokkos::create_mirror_view(surfaces);
    Kokkos::deep_copy(h_surfaces, surfaces);

    // Initialize to zeros
    for (size_t ns = 0; ns < nsurf; ++ns) {
        h_num_neighbors(ns) = 0;
        for (size_t n = 0; n < max_neighbors; ++n) {
            h_surface_neighbors(ns, n) = boundary;
        }
    }

    // Build connectivity: for each surface, find other surfaces sharing its endpoints
    for (size_t ns1 = 0; ns1 < nsurf; ++ns1) {
        size_t from1 = h_surfaces(ns1).from_node;
        size_t to1 = h_surfaces(ns1).to_node;

        size_t neighbor_count = 0;
        for (size_t ns2 = 0; ns2 < nsurf; ++ns2) {
            // if (ns1 == ns2) continue; // commenting out to see if this will add diagonals as well

            size_t from2 = h_surfaces(ns2).from_node;
            size_t to2 = h_surfaces(ns2).to_node;

            // Surfaces are neighbors if they share at least one channel
            if (from1 == from2 || from1 == to2 || to1 == from2 || to1 == to2) {
                h_surface_neighbors(ns1, neighbor_count) = ns2;
                neighbor_count++;
                if (neighbor_count >= max_neighbors) break;
            }
        }
        h_num_neighbors(ns1) = neighbor_count;
    }

    Kokkos::deep_copy(_surface_neighbors, h_surface_neighbors);
    Kokkos::deep_copy(_num_neighbors, h_num_neighbors);

    std::cout << "Surface connectivity built:" << std::endl;
    std::cout << "  Average neighbors per surface: "
              << std::accumulate(h_num_neighbors.data(), h_num_neighbors.data() + nsurf, 0.0) / nsurf
              << std::endl;
}

// Explicit template instantiations
template class Geometry<Kokkos::Serial>;
template class Geometry<Kokkos::OpenMP>;
#ifdef KOKKOS_ENABLE_CUDA
template class Geometry<Kokkos::Cuda>;
#endif