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

    // Build map from face/BC index to ray index using new ray structure
    for (size_t iray = 0; iray < _n_rays; iray++) {
        const auto& ray = _rays[iray];
        int ang = ray.angle();
        int bc_index = ray.bc_index(RAY_START);
        if (bc_index >= 0) {
            angface_to_ray[ang][ray.bc_face(RAY_START)][bc_index] = iray;
        }
        bc_index = ray.bc_index(RAY_END);
        if (bc_index >= 0) {
            angface_to_ray[ang][ray.bc_face(RAY_END)][bc_index] = _n_rays + iray;
        }
    }

    // Now allocate the angular flux arrays, remap the long ray indexes, and initialize the angular flux arrays
    total_bc_points = 2 * total_bc_points + 2;  // Both directions on each ray, plus two for the vacuum rays
    _h_angflux = HViewDouble3D("angflux", total_bc_points, _npol, _ng);
    _h_old_angflux = HViewDouble3D("old_angflux", total_bc_points, _npol, _ng);

    // Create device ray data structure
    auto h_ray_data = Kokkos::create_mirror_view(_d_ray_data);
    h_ray_data = Kokkos::View<DeviceRayData*, layout, Kokkos::HostSpace>("h_ray_data", _n_rays);
    _d_ray_data = DeviceRayView("d_ray_data", _n_rays);

    // Calculate segment start indices locally
    std::vector<int> ray_seg_starts(_n_rays + 1);
    ray_seg_starts[0] = 0;
    for (int iray = 0; iray < _n_rays; iray++) {
        ray_seg_starts[iray + 1] = ray_seg_starts[iray] + _rays[iray].nsegs();
    }
    int total_segments = ray_seg_starts[_n_rays];

    // Create device segment data structure
    auto h_segment_data = Kokkos::create_mirror_view(_d_segment_data);
    h_segment_data = Kokkos::View<DeviceSegmentData*, layout, Kokkos::HostSpace>("h_segment_data", total_segments);
    _d_segment_data = DeviceSegmentView("d_segment_data", total_segments);

    // Calculate BC indices and populate ray/segment data directly
    for (size_t i = 0; i < _n_rays; i++) {
        const auto& ray = _rays[i];
        int ang = ray.angle();
        int irefl = ang % 2 == 0 ? ang + 1 : ang - 1;

        // Calculate BC indices directly
        int bc_frwd_start, bc_frwd_end, bc_bkwd_start, bc_bkwd_end;

        if (ray.bc_index(RAY_START) == -1) {
            bc_frwd_start = total_bc_points - 2;
            bc_bkwd_end = total_bc_points - 1;
        } else {
            int start_index = ray.bc_index(RAY_START);
            bc_frwd_start = angface_to_ray[ang][ray.bc_face(RAY_START)][start_index];
            bc_bkwd_end = angface_to_ray[irefl][ray.bc_face(RAY_START)][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _h_angflux(bc_frwd_start, ipol, ig) = 0.0;
                    _h_angflux(bc_bkwd_end, ipol, ig) = 0.0;
                }
            }
        }

        if (ray.bc_index(RAY_END) == -1) {
            bc_bkwd_start = total_bc_points - 2;
            bc_frwd_end = total_bc_points - 1;
        } else {
            int start_index = ray.bc_index(RAY_END);
            bc_frwd_end = angface_to_ray[irefl][ray.bc_face(RAY_END)][start_index];
            bc_bkwd_start = angface_to_ray[ang][ray.bc_face(RAY_END)][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _h_angflux(bc_frwd_end, ipol, ig) = 0.0;
                    _h_angflux(bc_bkwd_start, ipol, ig) = 0.0;
                }
            }
        }

        // Populate ray data directly with calculated BC indices
        h_ray_data(i).nsegs = ray.nsegs();
        h_ray_data(i).angle = ray.angle();
        h_ray_data(i).seg_start = ray_seg_starts[i];
        h_ray_data(i).bc_frwd_start = bc_frwd_start;
        h_ray_data(i).bc_frwd_end = bc_frwd_end;
        h_ray_data(i).bc_bkwd_start = bc_bkwd_start;
        h_ray_data(i).bc_bkwd_end = bc_bkwd_end;

        // Set the processed BC indices in the ray object
        _rays[i].set_angflux_bc_indices(bc_frwd_start, bc_frwd_end, bc_bkwd_start, bc_bkwd_end);

        // Populate segment data for this ray
        int seg_start = ray_seg_starts[i];
        for (int iseg = 0; iseg < ray.nsegs(); iseg++) {
            h_segment_data(seg_start + iseg).fsr_id = ray.fsr(iseg) - 1; // Convert to 0-based
            h_segment_data(seg_start + iseg).length = ray.segment(iseg);
        }
    }

    Kokkos::deep_copy(_d_ray_data, h_ray_data);
    Kokkos::deep_copy(_d_segment_data, h_segment_data);

    // Store the inverse polar angle sine
    _h_rsinpolang = HViewDouble1D("rsinpolang", _npol);
    for (int ipol = 0; ipol < _npol; ipol++) {
        _h_rsinpolang(ipol) = 1.0 / std::sin(polar_angles[ipol]);
    }

    // Count maximum segments across all rays
    _max_segments = 0;
    for (const auto& ray : _rays) {
        _max_segments = std::max(_max_segments, ray.nsegs());
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
        _h_exp_table = HViewDouble2D("exp_table", _n_exp_intervals + 1, 2);
        double x1 = min_val;
        double y1 = 1.0 - Kokkos::exp(x1);
        for (size_t i = 0; i < _n_exp_intervals + 1; i++) {
            double x2 = x1 + dx;
            double y2 = 1.0 - Kokkos::exp(x2);
            _h_exp_table(i, 0) = (y2 - y1) * _exp_rdx;
            _h_exp_table(i, 1) = y1 - _h_exp_table(i, 0) * x1;
            x1 = x2;
            y1 = y2;
        }
        _d_exp_table = Kokkos::create_mirror(ExecutionSpace(), _h_exp_table);
        Kokkos::deep_copy(_d_exp_table, _h_exp_table);
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

    // Count the rays and set up the rays vector similar to SerialMOC
    _n_rays = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    _n_rays++;
                }
            }
        }
    }

    // Reserve space for rays
    _rays.reserve(_n_rays);

    // Set up the rays using KokkosLongRay
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);

            // Read the radians data from the angle group
            auto radians = angleGroup.getDataSet("Radians").read<double>();
            auto angleIndex = std::stoi(objName.substr(8)) - 1;

            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);
                    _rays.emplace_back(rayGroup, angleIndex, radians);
                }
            }
        }
    }

    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << _n_rays << " rays from file: " << _filename << std::endl;

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

template <typename ExecutionSpace>
KOKKOS_INLINE_FUNCTION
void compute_exparg(int iray, int ig, int ipol,
                    const Kokkos::View<const double**, ExecutionSpace>& exp_table,
                    int n_intervals, double rdx,
                    Kokkos::View<double*, typename ExecutionSpace::scratch_memory_space>& exparg,
                    const Kokkos::View<const double**, ExecutionSpace>& xstr,
                    const Kokkos::View<const int*, ExecutionSpace>& ray_fsrs,
                    const Kokkos::View<const double*, ExecutionSpace>& ray_segments,
                    const Kokkos::View<const double*, ExecutionSpace>& rsinpolang,
                    const Kokkos::View<const int*, ExecutionSpace>& ray_nsegs)
{
    for (int iseg = ray_nsegs(iray); iseg < ray_nsegs(iray + 1); iseg++) {
        int local_seg = iseg - ray_nsegs(iray);
        double val = -xstr(ray_fsrs(iseg), ig) * ray_segments(iseg) * rsinpolang(ipol);
        int i = Kokkos::floor(val * rdx) + n_intervals + 1;
        if (i >= 0 && i < n_intervals + 1) {
            exparg(local_seg) = exp_table(i, 0) * val + exp_table(i, 1);
        } else if (val < -700.0) {
            exparg(local_seg) = 1.0;
        } else {
            exparg(local_seg) = 1.0 - Kokkos::exp(val);
        }
    }
}

#ifdef KOKKOS_ENABLE_CUDA
template <>
KOKKOS_INLINE_FUNCTION
void compute_exparg<Kokkos::Cuda>(int iray, int ig, int ipol,
                    const Kokkos::View<const double**, Kokkos::Cuda>& exp_table,
                    int n_intervals, double rdx,
                    Kokkos::View<double*, typename Kokkos::Cuda::scratch_memory_space>& exparg,
                    const Kokkos::View<const double**, Kokkos::Cuda>& xstr,
                    const Kokkos::View<const int*, Kokkos::Cuda>& ray_fsrs,
                    const Kokkos::View<const double*, Kokkos::Cuda>& ray_segments,
                    const Kokkos::View<const double*, Kokkos::Cuda>& rsinpolang,
                    const Kokkos::View<const int*, Kokkos::Cuda>& ray_nsegs)
{
    return;
}
#endif

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

    // Avoid implicit capture
    auto& ray_data = _d_ray_data;
    auto& segment_data = _d_segment_data;
    auto& old_angflux = _d_old_angflux;
    auto& angflux = _d_angflux;
    auto& scalar_flux = _d_scalar_flux;
    auto& thread_scalar_flux = _d_thread_scalar_flux;
    auto& source = _d_source;
    auto& xstr = _d_xstr;
    auto& fsr_vol = _d_fsr_vol;
    auto& rsinpolang = _d_rsinpolang;
    auto& angle_weights = _d_angle_weights;

    // Copy old angular flux
    Kokkos::deep_copy(_d_old_angflux, _h_old_angflux);

    // Initialize the scalar flux to 0.0
    Kokkos::deep_copy(_d_scalar_flux, 0.0);

    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
        Kokkos::deep_copy(_d_thread_scalar_flux, 0.0);
    }

    // Sweep all rays using parallel_for with ray index
    Kokkos::parallel_for("MOC Sweep Rays",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>({0, 0, 0}, {_n_rays, _npol, _ng}),
        KOKKOS_LAMBDA(int iray, int ipol, int ig) {

            // Get ray data directly from DeviceRayData struct
            const auto& ray = ray_data(iray);
            int nsegs = ray.nsegs;
            int ray_angle = ray.angle;
            int seg_start = ray.seg_start;

            // Create temporary arrays for segment flux
            double fsegflux = old_angflux(ray.bc_frwd_start, ipol, ig);
            double bsegflux = old_angflux(ray.bc_bkwd_start, ipol, ig);

            // Forward and backward sweeps
            for (int iseg = 0; iseg < nsegs; iseg++) {
                // Forward segment sweep
                int global_seg = seg_start + iseg;
                const auto& segment = segment_data(global_seg);
                int ireg = segment.fsr_id;
                double segment_length = segment.length;
                double exp_arg = -xstr(ireg, ig) * segment_length * rsinpolang(ipol);
                double phid = (fsegflux - source(ireg, ig)) * (1.0 - Kokkos::exp(exp_arg));
                fsegflux -= phid;

                // Tally to scalar flux
                if constexpr(std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
                    thread_scalar_flux(omp_get_thread_num(), ireg, ig) +=
                        phid * angle_weights(ray_angle, ipol);
                } else {
                    Kokkos::atomic_add(&scalar_flux(ireg, ig),
                        phid * angle_weights(ray_angle, ipol));
                }

                // Backward segment sweep
                int bseg = nsegs - 1 - iseg;
                global_seg = seg_start + bseg;
                const auto& bsegment = segment_data(global_seg);
                ireg = bsegment.fsr_id;
                segment_length = bsegment.length;
                exp_arg = -xstr(ireg, ig) * segment_length * rsinpolang(ipol);
                phid = (bsegflux - source(ireg, ig)) * (1.0 - Kokkos::exp(exp_arg));
                bsegflux -= phid;

                // Tally to scalar flux
                if constexpr (std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
                    thread_scalar_flux(omp_get_thread_num(), ireg, ig) +=
                        phid * angle_weights(ray_angle, ipol);
                } else {
                    Kokkos::atomic_add(&scalar_flux(ireg, ig),
                        phid * angle_weights(ray_angle, ipol));
                }
            }

            // Store final segment flux back to angular flux arrays
            angflux(ray.bc_frwd_end, ipol, ig) = fsegflux;
            angflux(ray.bc_bkwd_end, ipol, ig) = bsegflux;
        }
    );

    // Reduction for OpenMP
    if constexpr(std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
        for (int k = 0; k < ExecutionSpace::concurrency(); k++) {
            for (int i = 0; i < _nfsr; i++) {
                for (int g = 0; g < _ng; g++) {
                    scalar_flux(i, g) += thread_scalar_flux(k, i, g);
                }
            }
        }
    }

    // Scale the flux with source, volume, and transport XS
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {_nfsr, _ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = scalar_flux(i, g) * _plane_height / (xstr(i, g) * fsr_vol(i)) + source(i, g) * fourpi;
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
