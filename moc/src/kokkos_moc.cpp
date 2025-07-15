#include "kokkos_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <cassert>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include "c5g7_library.hpp"

template <typename ExecutionSpace>
KokkosMOC<ExecutionSpace>::KokkosMOC(const ArgumentParser& args) :
    _filename(args.get_positional(0)),
    _file(HighFive::File(_filename, HighFive::File::ReadOnly)),
    _device(args.get_option("device"))
{
    // Read the rays
    _read_rays();

    Kokkos::Profiling::pushRegion("KokkosMOC::KokkosMOC init " + _device);
    // Read the FSR volumes and plane height
    {
        auto fsr_vol = _file.getDataSet("/MOC_Ray_Data/Domain_00001/FSR_Volume").read<std::vector<double>>();
        _nfsr = fsr_vol.size();
        _h_fsr_vol = HViewDouble1D("fsr_vol", _nfsr);
        for (int i = 0; i < _nfsr; i++){
            _h_fsr_vol(i) = fsr_vol[i];
        }
    }
    _plane_height = _file.getDataSet("/MOC_Ray_Data/Domain_00001/plane_height").read<double>();

    // Read mapping data
    auto xsrToFsrMap = _file.getDataSet("/MOC_Ray_Data/Domain_00001/XSRtoFSR_Map").read<std::vector<int>>();
    auto starting_xsr = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Starting XSR").read<int>();

    // Adjust xsrToFsrMap by subtracting starting_xsr from each element
    for (auto& xsr : xsrToFsrMap) {
        xsr -= starting_xsr;
    }

    // Read the material IDs
    auto tmp_mat_id = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xsr_mat_id").read<std::vector<double>>();
    std::vector<int> xsr_mat_id;
    xsr_mat_id.reserve(tmp_mat_id.size());
    for (const auto& id : tmp_mat_id) {
        xsr_mat_id.push_back(static_cast<int>(id) - 1);
    }

    // Calculate the FSR material IDs
    std::vector<int> fsr_mat_id(_nfsr);
    int ixsr = 0;
    for (int i = 0; i < _nfsr; i++) {
        if (i == xsrToFsrMap[ixsr]) {
            ixsr++;
        }
        fsr_mat_id[i] = xsr_mat_id[ixsr - 1];
    }

    // Get XS
    if (args.get_positional(0) == args.get_positional(1)) {
        auto xstr = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xstr").read<std::vector<std::vector<double>>>();
        auto xsnf = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xsnf").read<std::vector<std::vector<double>>>();
        auto xsch = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xsch").read<std::vector<std::vector<double>>>();
        auto xssc = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xssc").read<std::vector<std::vector<std::vector<double>>>>();
        _ng = xstr[0].size();
        _h_xstr = HViewDouble2D("xstr", _nfsr, _ng);
        _h_xsnf = HViewDouble2D("xsnf", _nfsr, _ng);
        _h_xsch = HViewDouble2D("xsch", _nfsr, _ng);
        _h_xssc = HViewDouble3D("xssc", _nfsr, _ng, _ng);
        int ixsr = 0;
        for (int i = 0; i < _nfsr; i++) {
            if (i == xsrToFsrMap[ixsr]) {
                ixsr++;
            }
            for (int to = 0; to < _ng; to++) {
                _h_xstr(i, to) = xstr[ixsr - 1][to];
                _h_xsnf(i, to) = xsnf[ixsr - 1][to];
                _h_xsch(i, to) = xsch[ixsr - 1][to];
                for (int from = 0; from < _ng; from++) {
                    _h_xssc(i, to, from) = xssc[ixsr - 1][to][from];
                }
            }
        }
    } else {
        auto library = c5g7_library(args.get_positional(1));
        _ng = library.get_num_groups();
        _get_xstr(_nfsr, fsr_mat_id, library);
        _get_xsnf(_nfsr, fsr_mat_id, library);
        _get_xsch(_nfsr, fsr_mat_id, library);
        _get_xssc(_nfsr, fsr_mat_id, library);
    }

    // Allocate scalar flux and source array
    _h_scalar_flux = HViewDouble2D("scalar_flux", _nfsr, _ng);
    _h_source = HViewDouble2D("source", _nfsr, _ng);
    Kokkos::deep_copy(_h_scalar_flux, 1.0);
    Kokkos::deep_copy(_h_source, 1.0);

    // Read ray spacings and angular flux BC dimensions
    auto domain = _file.getGroup("/MOC_Ray_Data/Domain_00001");
    auto polar_angles = _file.getDataSet("/MOC_Ray_Data/Polar_Radians").read<std::vector<double>>();
    auto polar_weights = _file.getDataSet("/MOC_Ray_Data/Polar_Weights").read<std::vector<double>>();
    auto azi_weights = _file.getDataSet("/MOC_Ray_Data/Azimuthal_Weights").read<std::vector<double>>();
    _npol = polar_angles.size();
    int nazi = azi_weights.size();
    std::vector<std::vector<int>> bc_sizes;
    int _max_bc_size = 0;
    int total_bc_points = 0;
    _ray_spacing.clear();
    for (const auto& objName : domain.listObjectNames()) {
        // Loop over each angle group
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            // Read ray spacing
            _ray_spacing.push_back(angleGroup.getDataSet("spacing").read<double>());  // Read the BC sizes
            int iazi = std::stoi(objName.substr(8)) - 1;
            std::vector<int> bc_size = angleGroup.getDataSet("BC_size").read<std::vector<int>>();
            bc_sizes.push_back(bc_size);
            _max_bc_size = std::max({_max_bc_size, bc_size[0], bc_size[1], bc_size[2], bc_size[3]});
            total_bc_points += std::accumulate(bc_size.begin(), bc_size.end(), 0);
        }
    }
    std::vector<std::vector<std::vector<int>>> angface_to_ray(nazi);
    for (int iazi = nazi - 1; iazi >= 0; iazi--) {
        angface_to_ray[iazi].resize(4);
        for (int iface = 3; iface >= 0; iface--) {
            angface_to_ray[iazi][iface].resize(bc_sizes[iazi][iface]);
            if (iazi == nazi - 1 && iface == 3) {
                bc_sizes[iazi][iface] = total_bc_points - bc_sizes[iazi][iface];
            } else if (iface == 3) {
                bc_sizes[iazi][iface] = bc_sizes[iazi + 1][0] - bc_sizes[iazi][iface];
            } else {
                bc_sizes[iazi][iface] = bc_sizes[iazi][iface + 1] - bc_sizes[iazi][iface];
            }
        }
    }

    // Build map from face/BC index to ray index
    for (size_t iray = 0; iray < _n_rays; iray++) {
        int ang = _h_ray_angle_index(iray);
        int bc_index = _h_ray_bc_index_frwd_start(iray);
        if (bc_index >= 0) {
            angface_to_ray[ang][_h_ray_bc_face_start(iray)][bc_index] = iray;
        }
        bc_index = _h_ray_bc_index_bkwd_start(iray);
        if (bc_index >= 0) {
            angface_to_ray[ang][_h_ray_bc_face_end(iray)][bc_index] = _n_rays + iray;
        }
    }

    // Now allocate the angular flux arrays, remap the long ray indexes, and initialize the angular flux arrays
    total_bc_points = 2 * total_bc_points + 2;  // Both directions on each ray, plus two for the vacuum rays
    _h_angflux = HViewDouble3D("angflux", total_bc_points, _npol, _ng);
    _h_old_angflux = HViewDouble3D("old_angflux", total_bc_points, _npol, _ng);
    for (size_t i = 0; i < _n_rays; i++) {
        int ang = _h_ray_angle_index(i);
        int irefl = ang % 2 == 0 ? ang + 1 : ang - 1;
        if (_h_ray_bc_index_frwd_start(i) == -1) {
            _h_ray_bc_index_frwd_start(i) = total_bc_points - 2;
            _h_ray_bc_index_bkwd_end(i) = total_bc_points - 1;
        } else {
            int start_index = _h_ray_bc_index_frwd_start(i);
            _h_ray_bc_index_frwd_start(i) = angface_to_ray[ang][_h_ray_bc_face_start(i)][start_index];
            _h_ray_bc_index_bkwd_end(i) = angface_to_ray[irefl][_h_ray_bc_face_start(i)][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _h_angflux(_h_ray_bc_index_frwd_start(i), ipol, ig) = 0.0;
                    _h_angflux(_h_ray_bc_index_bkwd_end(i), ipol, ig) = 0.0;
                }
            }
        }
        if (_h_ray_bc_index_frwd_end(i) == -1) {
            _h_ray_bc_index_bkwd_start(i) = total_bc_points - 2;
            _h_ray_bc_index_frwd_end(i) = total_bc_points - 1;
        } else {
            int start_index = _h_ray_bc_index_bkwd_start(i);
            _h_ray_bc_index_frwd_end(i) = angface_to_ray[irefl][_h_ray_bc_face_end(i)][start_index];
            _h_ray_bc_index_bkwd_start(i) = angface_to_ray[ang][_h_ray_bc_face_end(i)][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _h_angflux(_h_ray_bc_index_frwd_end(i), ipol, ig) = 0.0;
                    _h_angflux(_h_ray_bc_index_bkwd_start(i), ipol, ig) = 0.0;
                }
            }
        }
    }

    // Store the inverse polar angle sine
    _h_rsinpolang = HViewDouble1D("rsinpolang", _npol);
    for (int ipol = 0; ipol < _npol; ipol++) {
        _h_rsinpolang(ipol) = 1.0 / std::sin(polar_angles[ipol]);
    }

    // Count maximum segments across all rays
    _max_segments = 0;
    for (int i = 0; i < _n_rays; i++) {
        _max_segments = std::max(_max_segments, _h_ray_nsegs(i + 1) - _h_ray_nsegs(i));
    }
    Kokkos::deep_copy(_h_old_angflux, _h_angflux);

    // Build angle weights
    _h_angle_weights = HViewDouble2D("angle_weights", nazi, _npol);
    for (int iazi = 0; iazi < nazi; iazi++) {
        for (int ipol = 0; ipol < _npol; ipol++) {
            _h_angle_weights(iazi, ipol) = _ray_spacing[iazi] * azi_weights[iazi] * polar_weights[ipol]
                * M_PI * std::sin(polar_angles[ipol]);
        }
    }
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("KokkosMOC::KokkosMOC exp table " + _device);
    // Build exponential table (for all execution spaces that want to use table lookup)
    bool build_table = false;
#ifdef KOKKOS_ENABLE_SERIAL
    build_table = std::is_same_v<ExecutionSpace, Kokkos::Serial>;
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    build_table = std::is_same_v<ExecutionSpace, Kokkos::OpenMP> || build_table;
#endif
    if (build_table) {
        _n_exp_intervals = 40000;
        double min_val = -40.0;
        double max_val = 0.0;
        _exp_rdx = _n_exp_intervals / (max_val - min_val);
        double dx = 1.0 / _exp_rdx;

        // Build inline exponential tables for polar angle dependence
        _h_expoa = HViewDouble2D("expoa", _n_exp_intervals + 1, _npol);
        _h_expob = HViewDouble2D("expob", _n_exp_intervals + 1, _npol);

        for (int i = 0; i <= _n_exp_intervals; i++) {
            double x1 = min_val + i * dx;
            double x2 = x1 + dx;

            for (int ipol = 0; ipol < _npol; ipol++) {
                // Scale by polar angle sine factor like MPACT does
                double rpol = _h_rsinpolang(ipol);
                double x1_scaled = x1 * rpol;
                double x2_scaled = x2 * rpol;
                double y1_scaled = 1.0 - Kokkos::exp(x1_scaled);
                double y2_scaled = 1.0 - Kokkos::exp(x2_scaled);

                // Linear interpolation coefficients: y = m*x + b
                _h_expoa(i, ipol) = (y2_scaled - y1_scaled) / dx;  // slope
                _h_expob(i, ipol) = y1_scaled - _h_expoa(i, ipol) * x1;  // intercept
                _h_expoa(i, ipol) *= 0.001;  // Scale for pre-scaled segment lengths
            }
        }

        _d_expoa = Kokkos::create_mirror(ExecutionSpace(), _h_expoa);
        Kokkos::deep_copy(_d_expoa, _h_expoa);
        _d_expob = Kokkos::create_mirror(ExecutionSpace(), _h_expob);
        Kokkos::deep_copy(_d_expob, _h_expob);

        // Scale segment lengths by 1000.0 to avoid repeated multiplication in sweep
        for (size_t i = 0; i < _h_ray_segments.size(); i++) {
            _h_ray_segments(i) *= 1000.0;
        }
    }
    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("KokkosMOC::KokkosMOC mirror views " + _device);
    // Instead of conditional device setup, always initialize device views
    _d_angle_weights = Kokkos::create_mirror(ExecutionSpace(), _h_angle_weights);
    Kokkos::deep_copy(_d_angle_weights, _h_angle_weights);
    _d_fsr_vol = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_fsr_vol);
    _d_rsinpolang = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_rsinpolang);
    _d_xstr = Kokkos::create_mirror(ExecutionSpace(), _h_xstr);
    Kokkos::deep_copy(_d_xstr, _h_xstr);
    _d_xsnf = Kokkos::create_mirror(ExecutionSpace(), _h_xsnf);
    Kokkos::deep_copy(_d_xsnf, _h_xsnf);
    _d_xsch = Kokkos::create_mirror(ExecutionSpace(), _h_xsch);
    Kokkos::deep_copy(_d_xsch, _h_xsch);
    _d_xssc = Kokkos::create_mirror(ExecutionSpace(), _h_xssc);
    Kokkos::deep_copy(_d_xssc, _h_xssc);
    _d_scalar_flux = Kokkos::create_mirror(ExecutionSpace(), _h_scalar_flux);
    Kokkos::deep_copy(_d_scalar_flux, _h_scalar_flux);
    _d_source = Kokkos::create_mirror(ExecutionSpace(), _h_source);
    Kokkos::deep_copy(_d_source, _h_source);
    _d_ray_nsegs = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_nsegs);
    _d_ray_bc_face_start = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_bc_face_start);
    _d_ray_bc_face_end = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_bc_face_end);
    _d_ray_bc_index_frwd_start = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_bc_index_frwd_start);
    _d_ray_bc_index_frwd_end = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_bc_index_frwd_end);
    _d_ray_bc_index_bkwd_start = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_bc_index_bkwd_start);
    _d_ray_bc_index_bkwd_end = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_bc_index_bkwd_end);
    _d_ray_angle_index = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_angle_index);
    _d_ray_fsrs = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_fsrs);
    _d_ray_segments = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_ray_segments);
    _d_angflux = Kokkos::create_mirror(ExecutionSpace(), _h_angflux);
    Kokkos::deep_copy(_d_angflux, _h_angflux);
    _d_old_angflux = Kokkos::create_mirror(ExecutionSpace(), _h_old_angflux);
    Kokkos::deep_copy(_d_old_angflux, _h_old_angflux);

    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
        _d_thread_scalar_flux = DViewDouble3D("thread_scalar_flux", ExecutionSpace::concurrency(), _nfsr, _ng);
    }
    Kokkos::Profiling::popRegion();
}

// Implement other methods with template prefix
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_read_rays() {
    Kokkos::Profiling::pushRegion("KokkosMOC::KokkosMOC _read_rays " + _device);
    auto domain = _file.getGroup("/MOC_Ray_Data/Domain_00001");

    // Count the rays
    _n_rays = 0;
    for (size_t i = 0; i < domain.listObjectNames().size(); i++) {
        std::string objName = domain.listObjectNames()[i];
        if (objName.substr(0, 6) == "Angle_") {
            auto angleGroup = domain.getGroup(objName);
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    _n_rays++;
                }
            }
        }
    }

    // Convert the rays to a flattened format
    _h_ray_nsegs = HViewInt1D("ray_nsegs", _n_rays + 1);
    _h_ray_bc_face_start = HViewInt1D("ray_bc_face_start", _n_rays);
    _h_ray_bc_face_end = HViewInt1D("ray_bc_face_end", _n_rays);
    _h_ray_bc_index_frwd_start = HViewInt1D("ray_bc_index_frwd_start", _n_rays);
    _h_ray_bc_index_frwd_end = HViewInt1D("ray_bc_index_frwd_end", _n_rays);
    _h_ray_bc_index_bkwd_start = HViewInt1D("ray_bc_index_bkwd_start", _n_rays);
    _h_ray_bc_index_bkwd_end = HViewInt1D("ray_bc_index_bkwd_end", _n_rays);
    _h_ray_angle_index = HViewInt1D("ray_angle_index", _n_rays);
    _h_ray_nsegs(0) = 0;
    int iray = 0;
    int nangles = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);
                    auto fsrs = rayGroup.getDataSet("FSRs").read<std::vector<int>>();
                    _h_ray_nsegs(iray + 1) = _h_ray_nsegs(iray) + fsrs.size();
                    auto bcs = rayGroup.getDataSet("BC_face").read<std::vector<int>>();
                    _h_ray_bc_face_start(iray) = bcs[RAY_START] - 1;
                    _h_ray_bc_face_end(iray) = bcs[RAY_END] - 1;
                    bcs = rayGroup.getDataSet("BC_index").read<std::vector<int>>();
                    _h_ray_bc_index_frwd_start(iray) = bcs[RAY_START] - 1;
                    _h_ray_bc_index_frwd_end(iray) = bcs[RAY_END] - 1;
                    _h_ray_bc_index_bkwd_start(iray) = _h_ray_bc_index_frwd_end(iray);
                    _h_ray_bc_index_bkwd_end(iray) = _h_ray_bc_index_frwd_start(iray);
                    _h_ray_angle_index(iray) = nangles;
                    iray++;
                }
            }
            nangles++;
        }
    }

    _h_ray_fsrs = HViewInt1D("ray_fsrs", _h_ray_nsegs(_n_rays));
    _h_ray_segments = HViewDouble1D("ray_segments", _h_ray_nsegs(_n_rays));
    iray = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);
                    auto fsrs = rayGroup.getDataSet("FSRs").read<std::vector<int>>();
                    auto segments = rayGroup.getDataSet("Segments").read<std::vector<double>>();
                    for (size_t iseg = 0; iseg < fsrs.size(); iseg++) {
                        _h_ray_fsrs(_h_ray_nsegs(iray) + iseg) = fsrs[iseg] - 1; // Convert to zero-based index
                        _h_ray_segments(_h_ray_nsegs(iray) + iseg) = segments[iseg];
                    }
                    iray++;
                }
            }
        }
    }
    Kokkos::Profiling::popRegion();
}

// Get the total cross sections for each FSR from the library
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_get_xstr(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _h_xstr = HViewDouble2D("xstr", num_fsr, library.get_num_groups());
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        auto transport_xs = library.total(fsr_mat_id[i]);
        for (int g = 0; g < library.get_num_groups(); g++) {
            _h_xstr(i, g) = transport_xs[g];
        }
    }
}

// Get the nu-fission cross sections for each FSR from the library
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_get_xsnf(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _h_xsnf = HViewDouble2D("xsnf", num_fsr, library.get_num_groups());
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        auto nufiss_xs = library.nufiss(fsr_mat_id[i]);
        for (int g = 0; g < library.get_num_groups(); g++) {
            _h_xsnf(i, g) = nufiss_xs[g];
        }
    }
}

// Get the chi for each FSR from the library
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_get_xsch(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _h_xsch = HViewDouble2D("xsch", num_fsr, library.get_num_groups());
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        auto chi = library.chi(fsr_mat_id[i]);
        for (int g = 0; g < library.get_num_groups(); g++) {
            _h_xsch(i, g) = chi[g];
        }
    }
}

// Get the scattering cross sections for each FSR from the library
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_get_xssc(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _h_xssc = HViewDouble3D("xssc", num_fsr, library.get_num_groups(), library.get_num_groups());
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        for (int g = 0; g < library.get_num_groups(); g++) {
            for (int g2 = 0; g2 < library.get_num_groups(); g2++) {
                _h_xssc(i, g, g2) = library.scat(fsr_mat_id[i], g, g2);
            }
        }
    }
}

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
template <typename ExecutionSpace>
std::vector<double> KokkosMOC<ExecutionSpace>::fission_source(const double keff) const {
    Kokkos::Profiling::pushRegion("KokkosMOC::KokkosMOC fission source " + _device);
    std::vector<double> fissrc(_nfsr, 0.0);
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < _ng; g++) {
            fissrc[i] += _h_xsnf(i, g) * _h_scalar_flux(i, g) / keff;
        }
    }
    Kokkos::Profiling::popRegion();
    return fissrc;
}

// Build the total source term for each FSR based on the fission source and scattering cross sections
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::update_source(const std::vector<double>& fissrc) {
    Kokkos::Profiling::pushRegion("KokkosMOC::KokkosMOC update source " + _device);
    int _nfsr = _h_scalar_flux.extent(0);
    int ng = _h_scalar_flux.extent(1);
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < ng; g++) {
            _h_source(i, g) = fissrc[i] * _h_xsch(i, g);
            for (int g2 = 0; g2 < ng; g2++) {
                if (g != g2) {
                    _h_source(i, g) += _h_xssc(i, g, g2) * _h_scalar_flux(i, g2);
                }
            }
            _h_source(i, g) += _h_xssc(i, g, g) * _h_scalar_flux(i, g);
            _h_source(i, g) /= (_h_xstr(i, g) * fourpi);
        }
    }
    // Always copy to device
    Kokkos::deep_copy(_d_source, _h_source);
    Kokkos::Profiling::popRegion();
}

// General template implementation (fallback)
template <typename ExecutionSpace>
Kokkos::TeamPolicy<ExecutionSpace> KokkosMOC<ExecutionSpace>::_configure_team_policy(int n_rays, int npol, int ng) {
    // Default implementation for any unspecialized execution space
    const int n_teams = static_cast<long int>(n_rays) *  npol * ng;
    return Kokkos::TeamPolicy<ExecutionSpace>(n_teams, 1, 1);
}

#ifdef KOKKOS_ENABLE_CUDA
template <>
Kokkos::TeamPolicy<Kokkos::Cuda> KokkosMOC<Kokkos::Cuda>::_configure_team_policy(int n_rays, int npol, int ng) {
    const int n_teams = static_cast<long int>(n_rays);
    const int team_size = npol * ng;
    return Kokkos::TeamPolicy<Kokkos::Cuda>(n_teams, team_size, 1);
}
#endif

template <typename ExecSpace>
struct RayIndexCalculator {
    KOKKOS_INLINE_FUNCTION
    static void calculate(int league_rank, int team_rank,
                         int npol, int ng,
                         int& iray, int& ipol, int& ig) {
        // Default implementation
        int flat_idx = league_rank;
        iray = flat_idx / (npol * ng);
        ipol = (flat_idx / ng) % npol;
        ig = flat_idx % ng;
    }
};

// Specialization for Serial backend
#ifdef KOKKOS_ENABLE_CUDA
template <>
struct RayIndexCalculator<Kokkos::Cuda> {
    KOKKOS_INLINE_FUNCTION
    static void calculate(int league_rank, int team_rank,
                         int npol, int ng,
                         int& iray, int& ipol, int& ig) {
        iray = league_rank;
        ig = team_rank / npol;
        ipol = team_rank % npol;
    }
};
#endif

// Overloaded version for inline exponential tables (Serial and OpenMP)
template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void compute_exparg(int iray, int ig, int ipol,
                          const Kokkos::View<const double**, ExecutionSpace>& expoa,
                          const Kokkos::View<const double**, ExecutionSpace>& expob,
                          int n_intervals,
                          Kokkos::View<double*, typename ExecutionSpace::scratch_memory_space>& exparg,
                          const Kokkos::View<const double**, ExecutionSpace>& xstr,
                          const Kokkos::View<const int*, ExecutionSpace>& ray_fsrs,
                          const Kokkos::View<const double*, ExecutionSpace>& ray_segments,
                          const Kokkos::View<const int*, ExecutionSpace>& ray_nsegs)
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(!std::is_same_v<ExecutionSpace, Kokkos::Cuda>)
#endif
    {
        for (int iseg = ray_nsegs(iray); iseg < ray_nsegs(iray + 1); iseg++) {
            int local_seg = iseg - ray_nsegs(iray);
            double xval = -xstr(ray_fsrs(iseg), ig) * ray_segments(iseg);
            int ix = static_cast<int>(Kokkos::floor(xval)) + 40000;  // Scale to table index
            ix = Kokkos::fmax(ix, -40000);  // Clamp to table bounds
            ix = Kokkos::fmin(ix, 40000);
            exparg(local_seg) = expoa(ix, ipol) * xval + expob(ix, ipol);
        }
    }
}

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
double eval_exp_arg(const Kokkos::View<double*, typename ExecutionSpace::scratch_memory_space>& exparg,
                    const int iseg, const double xstr, const double ray_segment, const double rsinpolang)
{
    return exparg(iseg);
}

#ifdef KOKKOS_ENABLE_CUDA
template <>
KOKKOS_INLINE_FUNCTION
double eval_exp_arg<Kokkos::Cuda>(const Kokkos::View<double*, typename Kokkos::Cuda::scratch_memory_space>& exparg,
                    const int iseg, const double xstr, const double ray_segment, const double rsinpolang)
{
    return 1.0 - Kokkos::exp(-xstr * ray_segment * rsinpolang);
}
#endif

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void tally_scalar_flux(
    Kokkos::View<double**, typename ExecutionSpace::array_layout, typename ExecutionSpace::memory_space> flux,
    Kokkos::View<double***, typename ExecutionSpace::array_layout, typename ExecutionSpace::memory_space> threaded_flux,
    const int ireg,
    const int ig,
    const double contribution
) {
    Kokkos::atomic_add(&flux(ireg, ig), contribution);
}

#ifdef KOKKOS_ENABLE_SERIAL
template <>
KOKKOS_INLINE_FUNCTION
void tally_scalar_flux<Kokkos::Serial>(
    Kokkos::View<double**, typename Kokkos::Serial::array_layout, typename Kokkos::Serial::memory_space> flux,
    Kokkos::View<double***, typename Kokkos::Serial::array_layout, typename Kokkos::Serial::memory_space> threaded_flux,
    const int ireg,
    const int ig,
    const double contribution
) {
    flux(ireg, ig) += contribution;
}
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template <>
KOKKOS_INLINE_FUNCTION
void tally_scalar_flux<Kokkos::OpenMP>(
    Kokkos::View<double**, typename Kokkos::OpenMP::array_layout, typename Kokkos::OpenMP::memory_space> flux,
    Kokkos::View<double***, typename Kokkos::OpenMP::array_layout, typename Kokkos::OpenMP::memory_space> threaded_flux,
    const int ireg,
    const int ig,
    const double contribution
) {
    threaded_flux(omp_get_thread_num(), ireg, ig) += contribution;
}
#endif

// Unified implementation of sweep
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::sweep() {
    Kokkos::Profiling::pushRegion("KokkosMOC::Sweep " + _device);
    using ScratchViewDouble1D = Kokkos::View<double*, typename ExecutionSpace::scratch_memory_space>;
    Kokkos::deep_copy(_d_old_angflux, _h_old_angflux);

    auto& scalar_flux = _d_scalar_flux;
    auto& ray_nsegs = _d_ray_nsegs;
    auto& ray_fsrs = _d_ray_fsrs;
    auto& ray_segments = _d_ray_segments;
    auto& ray_bc_index_frwd_start = _d_ray_bc_index_frwd_start;
    auto& ray_bc_index_frwd_end = _d_ray_bc_index_frwd_end;
    auto& ray_bc_index_bkwd_start = _d_ray_bc_index_bkwd_start;
    auto& ray_bc_index_bkwd_end = _d_ray_bc_index_bkwd_end;
    auto& ray_angle_index = _d_ray_angle_index;
    auto& n_rays = _n_rays;
    auto& npol = _npol;
    auto& nfsr = _nfsr;
    auto& ng = _ng;
    auto& xstr = _d_xstr;
    auto& source = _d_source;
    auto& fsr_vol = _d_fsr_vol;
    auto& dz = _plane_height;
    auto& rsinpolang = _d_rsinpolang;
    auto& angle_weights = _d_angle_weights;
    auto& angflux = _d_angflux;
    auto& old_angflux = _d_old_angflux;
    auto& max_segments = _max_segments;
    auto& n_exp_intervals = _n_exp_intervals;
    auto& expoa = _d_expoa;
    auto& expob = _d_expob;
    auto& threaded_scalar_flux = _d_thread_scalar_flux;

    // Initialize the scalar flux to 0.0
    Kokkos::deep_copy(scalar_flux, 0.0);
    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
        Kokkos::deep_copy(threaded_scalar_flux, 0.0);
    }

    // Use the specialized team policy configuration
    typedef typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member;
    auto policy = _configure_team_policy(n_rays, npol, ng);

    // Add scratch space for exponential arguments
    if (_device != "cuda") {
        size_t scratch_size = ScratchViewDouble1D::shmem_size(max_segments);
        policy = policy.set_scratch_size(0, Kokkos::PerTeam(scratch_size));
    }

    // Use the team-based approach for all backends
    Kokkos::parallel_for("MOC Sweep Rays", policy, KOKKOS_LAMBDA(const team_member& teamMember) {
        int iray, ipol, ig;

        // Use the specialized helper to calculate indices - no branches!
        RayIndexCalculator<ExecutionSpace>::calculate(
            teamMember.league_rank(), teamMember.team_rank(),
            npol, ng, iray, ipol, ig);

        int nsegs = ray_nsegs(iray + 1) - ray_nsegs(iray);

        // Create thread-local exparg array for non-CUDA execution spaces using scratch space
        ScratchViewDouble1D exparg(teamMember.team_scratch(0), nsegs);

        // Use inline exponential tables for Serial and OpenMP, regular tables for others
        compute_exparg<ExecutionSpace>(iray, ig, ipol, expoa, expob, n_exp_intervals, exparg,
                                       xstr, ray_fsrs, ray_segments, ray_nsegs);

        int global_seg, ireg1, ireg2;
        int iseg2 = nsegs;
        double phid;

        double fsegflux = old_angflux(ray_bc_index_frwd_start(iray), ipol, ig);
        double bsegflux = old_angflux(ray_bc_index_bkwd_start(iray), ipol, ig);

        // Rest of the sweep implementation remains the same
        for (int iseg1 = 0; iseg1 < nsegs; iseg1++) {
            // Forward segment sweep
            global_seg = ray_nsegs(iray) + iseg1;
            ireg1 = ray_fsrs(global_seg);
            phid = (fsegflux - source(ireg1, ig)) *
                   eval_exp_arg<ExecutionSpace>(exparg, iseg1, xstr(ireg1, ig),
                                                ray_segments(global_seg), rsinpolang(ipol));
            fsegflux -= phid;
            tally_scalar_flux<ExecutionSpace>(
                scalar_flux,
                threaded_scalar_flux,
                ireg1,
                ig,
                phid * angle_weights(ray_angle_index(iray), ipol)
            );

            // Backward segment sweep
            global_seg = ray_nsegs(iray) + iseg2 - 1;
            ireg2 = ray_fsrs(global_seg);
            phid = (bsegflux - source(ireg2, ig)) *
                   eval_exp_arg<ExecutionSpace>(exparg, iseg2 - 1, xstr(ireg2, ig),
                                                ray_segments(global_seg), rsinpolang(ipol));
            bsegflux -= phid;
            tally_scalar_flux<ExecutionSpace>(
                scalar_flux,
                threaded_scalar_flux,
                ireg2,
                ig,
                phid * angle_weights(ray_angle_index(iray), ipol)
            );
            iseg2--;
        }

        angflux(ray_bc_index_frwd_end(iray), ipol, ig) = fsegflux;
        angflux(ray_bc_index_bkwd_end(iray), ipol, ig) = bsegflux;
    });

    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
        for (int k = 0; k < ExecutionSpace::concurrency(); k++) {
            for (int i = 0; i < nfsr; i++) {
                for (int g = 0; g < ng; g++) {
                    scalar_flux(i, g) += threaded_scalar_flux(k, i, g);
                }
            }
        }
    }

    // Scale the flux with source, volume, and transport XS
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = scalar_flux(i, g) * dz / (xstr(i, g) * fsr_vol(i)) + source(i, g) * fourpi;
    });

    // Copy results back to host
    Kokkos::deep_copy(_h_scalar_flux, _d_scalar_flux);
    Kokkos::deep_copy(_h_angflux, _d_angflux);
    Kokkos::deep_copy(_h_old_angflux, _h_angflux);
    Kokkos::Profiling::popRegion();
}

// Explicit template instantiations
#ifdef KOKKOS_ENABLE_SERIAL
template class KokkosMOC<Kokkos::Serial>;
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template class KokkosMOC<Kokkos::OpenMP>;
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class KokkosMOC<Kokkos::Cuda>;
#endif
