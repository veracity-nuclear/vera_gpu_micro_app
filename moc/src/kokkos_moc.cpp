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
    _library(args.get_positional(1)),
    _file(HighFive::File(_filename, HighFive::File::ReadOnly)),
    _device(args.get_option("device"))
{
    // Read the rays
    _read_rays();

    // Read mapping data
    auto xsrToFsrMap = _file.getDataSet("/MOC_Ray_Data/Domain_00001/XSRtoFSR_Map").read<std::vector<int>>();
    auto starting_xsr = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Starting XSR").read<int>();

    // Adjust xsrToFsrMap by subtracting starting_xsr from each element
    for (auto& xsr : xsrToFsrMap) {
        xsr -= starting_xsr;
    }

    // Read the FSR volumes and plane height
    {
        auto fsr_vol = std::make_unique<std::vector<double>>();
        _file.getDataSet("/MOC_Ray_Data/Domain_00001/FSR_Volume").read(*fsr_vol);
        _nfsr = fsr_vol->size();
        _h_fsr_vol = Kokkos::View<double*, layout, Kokkos::HostSpace>("fsr_vol", _nfsr);
        for (int i = 0; i < _nfsr; i++){
            _h_fsr_vol(i) = (*fsr_vol)[i];
        }
    }
    _plane_height = _file.getDataSet("/MOC_Ray_Data/Domain_00001/plane_height").read<double>();

    // Initialize the library
    _ng = _library.get_num_groups();

    // Read the material IDs
    auto tmp_mat_id = _file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xsr_mat_id").read<std::vector<double>>();
    std::vector<int> xsr_mat_id;
    xsr_mat_id.reserve(tmp_mat_id.size());
    for (const auto& id : tmp_mat_id) {
        xsr_mat_id.push_back(static_cast<int>(id) - 1);
    }

    // Calculate the FSR material IDs
    _fsr_mat_id.resize(_nfsr);
    int ixsr = 0;
    for (int i = 0; i < _nfsr; i++) {
        if (i == xsrToFsrMap[ixsr]) {
            ixsr++;
        }
        _fsr_mat_id[i] = xsr_mat_id[ixsr - 1];
    }

    // Get XS
    _get_xstr(starting_xsr);

    // Allocate scalar flux and source array
    _h_scalar_flux = Kokkos::View<double**, layout, Kokkos::HostSpace>("scalar_flux", _nfsr, _ng);
    _h_source = Kokkos::View<double**, layout, Kokkos::HostSpace>("source", _nfsr, _ng);
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
            _ray_spacing.push_back(angleGroup.getDataSet("spacing").read<double>());            // Read the BC sizes
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
        if (_h_ray_bc_index_frwd_start(iray) >= 0) {
            angface_to_ray[_h_ray_angle_index(iray)][_h_ray_bc_face_start(iray)][_h_ray_bc_index_frwd_start(iray)] = iray;
        }
        if (_h_ray_bc_index_bkwd_start(iray) >= 0) {
            angface_to_ray[_h_ray_angle_index(iray)][_h_ray_bc_face_end(iray)][_h_ray_bc_index_bkwd_start(iray)] = _n_rays + iray;
        }
    }

    // Now allocate the angular flux arrays, remap the long ray indexes, and initialize the angular flux arrays
    total_bc_points = 2 * total_bc_points + 2;  // Both directions on each ray, plus two for the vacuum rays
    _h_angflux = Kokkos::View<double***, layout, Kokkos::HostSpace>("angflux", total_bc_points, _npol, _ng);
    _h_old_angflux = Kokkos::View<double***, layout, Kokkos::HostSpace>("old_angflux", total_bc_points, _npol, _ng);
    for (size_t i = 0; i < _n_rays; i++) {
        int irefl = _h_ray_angle_index(i) % 2 == 0 ? _h_ray_angle_index(i) + 1 : _h_ray_angle_index(i) - 1;
        if (_h_ray_bc_index_frwd_start(i) == -1) {
            _h_ray_bc_index_frwd_start(i) = total_bc_points - 2;
            _h_ray_bc_index_bkwd_end(i) = total_bc_points - 1;
        } else {
            int start_index = _h_ray_bc_index_frwd_start(i);
            _h_ray_bc_index_frwd_start(i) = angface_to_ray[_h_ray_angle_index(i)][_h_ray_bc_face_start(i)][start_index];
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
            _h_ray_bc_index_bkwd_start(i) = angface_to_ray[_h_ray_angle_index(i)][_h_ray_bc_face_end(i)][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _h_angflux(_h_ray_bc_index_frwd_end(i), ipol, ig) = 0.0;
                    _h_angflux(_h_ray_bc_index_bkwd_start(i), ipol, ig) = 0.0;
                }
            }
        }
    }
    Kokkos::deep_copy(_h_old_angflux, _h_angflux);

    // Build angle weights
    _h_angle_weights = Kokkos::View<double**, layout, Kokkos::HostSpace>("angle_weights", nazi, _npol);
    for (int iazi = 0; iazi < nazi; iazi++) {
        for (int ipol = 0; ipol < _npol; ipol++) {
            _h_angle_weights(iazi, ipol) = _ray_spacing[iazi] * azi_weights[iazi] * polar_weights[ipol]
                * M_PI * std::sin(polar_angles[ipol]);
        }
    }

    // Store the inverse polar angle sine
    _h_rsinpolang = Kokkos::View<double*, layout, Kokkos::HostSpace>("rsinpolang", _npol);
    for (int ipol = 0; ipol < _npol; ipol++) {
        _h_rsinpolang(ipol) = 1.0 / std::sin(polar_angles[ipol]);
    }

    // Count maximum segments across all rays
    _max_segments = 0;
    for (int i = 0; i < _n_rays; i++) {
        _max_segments = std::max(_max_segments, _h_ray_nsegs(i + 1) - _h_ray_nsegs(i));
    }

    // Allocate arrays needed during serial sweep
    if (_device == "serial") {
        _h_exparg = Kokkos::View<double**, Kokkos::HostSpace>("exparg", _max_segments + 1, _ng);
    }

    // Instead of conditional device setup, always initialize device views
    _d_angle_weights = Kokkos::create_mirror(ExecutionSpace(), _h_angle_weights);
    Kokkos::deep_copy(_d_angle_weights, _h_angle_weights);
    _d_fsr_vol = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_fsr_vol);
    _d_rsinpolang = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_rsinpolang);
    _d_xstr = Kokkos::create_mirror(ExecutionSpace(), _h_xstr);
    Kokkos::deep_copy(_d_xstr, _h_xstr);
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
}

// Implement other methods with template prefix
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_read_rays() {
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
    _h_ray_nsegs = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_nsegs", _n_rays + 1);
    _h_ray_bc_face_start = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_bc_face_start", _n_rays);
    _h_ray_bc_face_end = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_bc_face_end", _n_rays);
    _h_ray_bc_index_frwd_start = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_bc_index_frwd_start", _n_rays);
    _h_ray_bc_index_frwd_end = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_bc_index_frwd_end", _n_rays);
    _h_ray_bc_index_bkwd_start = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_bc_index_bkwd_start", _n_rays);
    _h_ray_bc_index_bkwd_end = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_bc_index_bkwd_end", _n_rays);
    _h_ray_angle_index = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_angle_index", _n_rays);
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

    _h_ray_fsrs = Kokkos::View<int*, layout, Kokkos::HostSpace>("ray_fsrs", _h_ray_nsegs(_n_rays));
    _h_ray_segments = Kokkos::View<double*, layout, Kokkos::HostSpace>("ray_segments", _h_ray_nsegs(_n_rays));
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
}

// Get the total cross sections for each FSR from the library
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_get_xstr(
    const int starting_xsr
) {
    _h_xstr = Kokkos::View<double**, layout, Kokkos::HostSpace>("xstr", _nfsr, _library.get_num_groups());
    for (auto i = 0; i < _fsr_mat_id.size(); i++) {
        auto total_xs = _library.total(_fsr_mat_id[i]);
        for (int g = 0; g < _library.get_num_groups(); g++) {
            _h_xstr(i, g) = total_xs[g];
        }
    }
}

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
template <typename ExecutionSpace>
std::vector<double> KokkosMOC<ExecutionSpace>::fission_source(const double keff) const {
    std::vector<double> fissrc(_nfsr, 0.0);
    for (size_t i = 0; i < _nfsr; i++) {
        if (_library.is_fissile(_fsr_mat_id[i])) {
            for (int g = 0; g < _ng; g++) {
                fissrc[i] += _library.nufiss(_fsr_mat_id[i], g) * _h_scalar_flux(i, g) / keff;
            }
        }
    }
    return fissrc;
}

// Build the total source term for each FSR based on the fission source and scattering cross sections
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::update_source(const std::vector<double>& fissrc) {
    int _nfsr = _h_scalar_flux.extent(0);
    int ng = _h_scalar_flux.extent(1);
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < ng; g++) {
            _h_source(i, g) = fissrc[i] * _library.chi(_fsr_mat_id[i], g);
            for (int g2 = 0; g2 < ng; g2++) {
                if (g != g2) {
                    _h_source(i, g) += _library.scat(_fsr_mat_id[i], g, g2) * _h_scalar_flux(i, g2);
                }
            }
            _h_source(i, g) += _library.self_scat(_fsr_mat_id[i], g) * _h_scalar_flux(i, g);
            _h_source(i, g) /= (_library.total(_fsr_mat_id[i], g) * 4.0 * M_PI);
        }
    }
    // Always copy to device
    Kokkos::deep_copy(_d_source, _h_source);
}

// General template implementation (fallback)
template <typename ExecutionSpace>
Kokkos::TeamPolicy<ExecutionSpace> KokkosMOC<ExecutionSpace>::_configure_team_policy(int n_rays, int npol, int ng) {
    // Default implementation for any unspecialized execution space
    const int n_teams = static_cast<long int>(n_rays) *  npol * ng;
    return Kokkos::TeamPolicy<ExecutionSpace>(n_teams, 1, 1);
}

// Specialization for CUDA
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
        // Default implementation (for CUDA and others)
        iray = league_rank;
        ig = team_rank / npol;
        ipol = team_rank % npol;
    }
};

// Specialization for Serial backend
#ifdef KOKKOS_ENABLE_SERIAL
template <>
struct RayIndexCalculator<Kokkos::Serial> {
    KOKKOS_INLINE_FUNCTION
    static void calculate(int league_rank, int team_rank,
                         int npol, int ng,
                         int& iray, int& ipol, int& ig) {
        int flat_idx = league_rank;
        iray = flat_idx / (npol * ng);
        ipol = (flat_idx / ng) % npol;
        ig = flat_idx % ng;
    }
};
#endif

// Specialization for OpenMP backend
#ifdef KOKKOS_ENABLE_OPENMP
template <>
struct RayIndexCalculator<Kokkos::OpenMP> {
    KOKKOS_INLINE_FUNCTION
    static void calculate(int league_rank, int team_rank,
                         int npol, int ng,
                         int& iray, int& ipol, int& ig) {
        int flat_idx = league_rank;
        iray = flat_idx / (npol * ng);
        ipol = (flat_idx / ng) % npol;
        ig = flat_idx % ng;
    }
};
#endif

// Unified implementation of sweep
template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::_impl_sweep() {
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

    // Initialize the scalar flux to 0.0
    Kokkos::deep_copy(scalar_flux, 0.0);

    // Use the specialized team policy configuration
    typedef typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member;
    auto policy = _configure_team_policy(n_rays, npol, ng);

    // Use the team-based approach for all backends
    Kokkos::parallel_for("MOC Sweep Rays", policy, KOKKOS_LAMBDA(const team_member& teamMember) {
        int iray, ipol, ig;

        // Use the specialized helper to calculate indices - no branches!
        RayIndexCalculator<ExecutionSpace>::calculate(
            teamMember.league_rank(), teamMember.team_rank(),
            npol, ng, iray, ipol, ig);

        int nsegs = ray_nsegs(iray + 1) - ray_nsegs(iray);

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
            phid = (fsegflux - source(ireg1, ig)) * (1.0 - exp(-xstr(ireg1, ig) * ray_segments(global_seg) * rsinpolang(ipol)));
            fsegflux -= phid;
            Kokkos::atomic_add(&scalar_flux(ireg1, ig), phid * angle_weights(ray_angle_index(iray), ipol));

            // Backward segment sweep
            global_seg = ray_nsegs(iray) + iseg2 - 1;
            ireg2 = ray_fsrs(global_seg);
            phid = (bsegflux - source(ireg2, ig)) * (1.0 - exp(-xstr(ireg2, ig) * ray_segments(global_seg) * rsinpolang(ipol)));
            bsegflux -= phid;
            Kokkos::atomic_add(&scalar_flux(ireg2, ig), phid * angle_weights(ray_angle_index(iray), ipol));
            iseg2--;
        }

        angflux(ray_bc_index_frwd_end(iray), ipol, ig) = fsegflux;
        angflux(ray_bc_index_bkwd_end(iray), ipol, ig) = bsegflux;
    });

    // Scale the flux with source, volume, and transport XS
    constexpr double fourpi = 4.0 * M_PI;
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = scalar_flux(i, g) * dz / (xstr(i, g) * fsr_vol(i)) + source(i, g) * fourpi;
    });

    // Copy results back to host
    Kokkos::deep_copy(_h_scalar_flux, _d_scalar_flux);
    Kokkos::deep_copy(_h_angflux, _d_angflux);
    Kokkos::deep_copy(_h_old_angflux, _h_angflux);
}

template <typename ExecutionSpace>
void KokkosMOC<ExecutionSpace>::sweep() {
    _impl_sweep();
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
