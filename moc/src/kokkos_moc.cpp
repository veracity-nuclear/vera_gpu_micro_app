#include "kokkos_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <cassert>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include "c5g7_library.hpp"

KokkosMOC::KokkosMOC(const ArgumentParser& args) :
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
        _h_fsr_vol = Kokkos::View<double*, Kokkos::HostSpace>("fsr_vol", _nfsr);
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
    _h_scalar_flux = Kokkos::View<double**, Kokkos::HostSpace>("scalar_flux", _nfsr, _ng);
    _h_source = Kokkos::View<double**, Kokkos::HostSpace>("source", _nfsr, _ng);
    for (size_t i = 0; i < _nfsr; ++i) {
        for (size_t j = 0; j < _ng; ++j) {
            _h_scalar_flux(i, j) = 1.0;
            _h_source(i, j) = 1.0;
        }
    }

    // Read ray spacings and angular flux BC dimensions
    auto domain = _file.getGroup("/MOC_Ray_Data/Domain_00001");
    auto polar_angles = _file.getDataSet("/MOC_Ray_Data/Polar_Radians").read<std::vector<double>>();
    auto polar_weights = _file.getDataSet("/MOC_Ray_Data/Polar_Weights").read<std::vector<double>>();
    auto azi_weights = _file.getDataSet("/MOC_Ray_Data/Azimuthal_Weights").read<std::vector<double>>();
    _npol = polar_angles.size();
    int nazi = 0;
    std::vector<std::vector<int>> bc_sizes;
    int _max_bc_size = 0;
    int total_bc_points = 0;
    _ray_spacing.clear();    for (const auto& objName : domain.listObjectNames()) {
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
            nazi++;
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
    _h_angflux = Kokkos::View<double***, Kokkos::HostSpace>("angflux", total_bc_points, _npol, _ng);
    _h_old_angflux = Kokkos::View<double***, Kokkos::HostSpace>("old_angflux", total_bc_points, _npol, _ng);
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
    _h_angle_weights = Kokkos::View<double**, Kokkos::HostSpace>("angle_weights", nazi, _npol);
    for (int iazi = 0; iazi < nazi; iazi++) {
        for (int ipol = 0; ipol < _npol; ipol++) {
            _h_angle_weights(iazi, ipol) = _ray_spacing[iazi] * azi_weights[iazi] * polar_weights[ipol]
                * M_PI * std::sin(polar_angles[ipol]);
        }
    }

    // Store the inverse polar angle sine
    _h_rsinpolang = Kokkos::View<double*, Kokkos::HostSpace>("rsinpolang", _npol);
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
        _h_segflux = Kokkos::View<double***, Kokkos::HostSpace>("segflux", 2, _max_segments + 1, _ng);
        _h_exparg = Kokkos::View<double**, Kokkos::HostSpace>("exparg", _max_segments + 1, _ng);
    }

    // Set up device views as needed
    if (_device == "cuda") {
        _d_fsr_vol = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), _h_fsr_vol);
        _d_rsinpolang = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), _h_rsinpolang);
        Kokkos::deep_copy(_d_xstr, _h_xstr);
        Kokkos::deep_copy(_d_scalar_flux, _h_scalar_flux);
        Kokkos::deep_copy(_d_source, _h_source);
    }
}

void KokkosMOC::_read_rays() {
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
    _h_ray_nsegs = Kokkos::View<int*, Kokkos::HostSpace>("ray_nsegs", _n_rays + 1);
    _h_ray_bc_face_start = Kokkos::View<int*, Kokkos::HostSpace>("ray_bc_face_start", _n_rays);
    _h_ray_bc_face_end = Kokkos::View<int*, Kokkos::HostSpace>("ray_bc_face_end", _n_rays);
    _h_ray_bc_index_frwd_start = Kokkos::View<int*, Kokkos::HostSpace>("ray_bc_index_frwd_start", _n_rays);
    _h_ray_bc_index_frwd_end = Kokkos::View<int*, Kokkos::HostSpace>("ray_bc_index_frwd_end", _n_rays);
    _h_ray_bc_index_bkwd_start = Kokkos::View<int*, Kokkos::HostSpace>("ray_bc_index_bkwd_start", _n_rays);
    _h_ray_bc_index_bkwd_end = Kokkos::View<int*, Kokkos::HostSpace>("ray_bc_index_bkwd_end", _n_rays);
    _h_ray_angle_index = Kokkos::View<int*, Kokkos::HostSpace>("ray_angle_index", _n_rays);
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

    _h_ray_fsrs = Kokkos::View<int*, Kokkos::HostSpace>("ray_fsrs", _h_ray_nsegs(_n_rays));
    _h_ray_segments = Kokkos::View<double*, Kokkos::HostSpace>("ray_segments", _h_ray_nsegs(_n_rays));
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
void KokkosMOC::_get_xstr(
    const int starting_xsr
) {
    _h_xstr = Kokkos::View<double**, Kokkos::HostSpace>("xstr", _nfsr, _library.get_num_groups());
    for (auto i = 0; i < _fsr_mat_id.size(); i++) {
        auto total_xs = _library.total(_fsr_mat_id[i]);
        for (int g = 0; g < _library.get_num_groups(); g++) {
            _h_xstr(i, g) = total_xs[g];
        }
    }
}

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
std::vector<double> KokkosMOC::fission_source(const double keff) const {
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
void KokkosMOC::update_source(const std::vector<double>& fissrc) {
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
    if (_device == "cuda") {
        Kokkos::deep_copy(_d_source, _h_source);
    }
}

void KokkosMOC::_impl_sweep_openmp() {
    using MemSpace = Kokkos::OpenMP;
    using ExecSpace = MemSpace::execution_space;

    auto& scalar_flux = _h_scalar_flux;
    auto& ray_nsegs = _h_ray_nsegs;
    auto& ray_fsrs = _h_ray_fsrs;
    auto& ray_segments = _h_ray_segments;
    auto& ray_bc_index_frwd_start = _h_ray_bc_index_frwd_start;
    auto& ray_bc_index_frwd_end = _h_ray_bc_index_frwd_end;
    auto& ray_bc_index_bkwd_start = _h_ray_bc_index_bkwd_start;
    auto& ray_bc_index_bkwd_end = _h_ray_bc_index_bkwd_end;
    auto& ray_angle_index = _h_ray_angle_index;
    auto& n_rays = _n_rays;
    auto& npol = _npol;
    auto& nfsr = _nfsr;
    auto& ng = _ng;
    auto& max_segments = _max_segments;
    auto& xstr = _h_xstr;
    auto& source = _h_source;
    auto& fsr_vol = _h_fsr_vol;
    auto& dz = _plane_height;
    auto& rsinpolang = _h_rsinpolang;
    auto& angle_weights = _h_angle_weights;
    auto& angflux = _h_angflux;
    auto& old_angflux = _h_old_angflux;

    // Initialize the scalar flux to 0.0; use thread-local flux to prevent need for atomic_add that
    // kills OpenMP performance
    int nthreads = Kokkos::OpenMP::concurrency();
    Kokkos::View<double***, Kokkos::HostSpace> thread_scalar_flux("thread scalar flux", nthreads, nfsr, ng);
    Kokkos::parallel_for("InitializeScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {nfsr, ng, nthreads}),
        KOKKOS_LAMBDA(int i, int j, int k) {
            scalar_flux(i, j) = 0.0;
	    thread_scalar_flux(k, i, j) = 0.0;
    });

    // Prepare scratch space
    typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
    typedef typename team_policy::member_type team_member;
    team_policy policy(static_cast<long int>(n_rays) * npol * ng, Kokkos::AUTO, Kokkos::AUTO);
    const size_t bytes_needed_per_team =
        Kokkos::View<double*, typename team_member::scratch_memory_space>::shmem_size(max_segments + 1) // exparg
        + 2 * Kokkos::View<double*, typename team_member::scratch_memory_space>::shmem_size(max_segments + 1); // segflux
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes_needed_per_team));

    // Sweep all the long rays
    Kokkos::parallel_for("Sweep Rays", policy, KOKKOS_LAMBDA(const team_member& teamMember) {
        int thread = omp_get_thread_num(); // Not portable, but works if we're using OpenMP
        int iray = teamMember.league_rank() / (npol * ng);
        int ipol = (teamMember.league_rank() % (npol * ng)) / ng;
        int ig = teamMember.league_rank() % ng;
        int nsegs = ray_nsegs(iray + 1) - ray_nsegs(iray);
        Kokkos::View<double*, typename team_member::scratch_memory_space> exparg(teamMember.team_scratch(0), nsegs);
        Kokkos::View<double**, typename team_member::scratch_memory_space> segflux(teamMember.team_scratch(0), 2, nsegs + 1);

        // Allocate and initialize exparg with dimensions [ray._nsegs][ng]
        for (int j = ray_nsegs(iray); j < ray_nsegs(iray + 1); j++) {
            exparg(j - ray_nsegs(iray)) = 1.0 - exp(-xstr(ray_fsrs(j), ig) * ray_segments(j) * rsinpolang(ipol));
        }

        int ireg1, ireg2;
        int iseg2 = nsegs;
        double phid;

        segflux(RAY_START, 0) = old_angflux(ray_bc_index_frwd_start(iray), ipol, ig);
        segflux(RAY_END, nsegs) = old_angflux(ray_bc_index_bkwd_start(iray), ipol, ig);
        for (int iseg1 = 0; iseg1 < nsegs; iseg1++) {
            // Forward segment sweep
            ireg1 = ray_fsrs(ray_nsegs(iray) + iseg1);
            phid = (segflux(RAY_START, iseg1) - source(ireg1, ig)) * exparg(iseg1);
            segflux(RAY_START, iseg1 + 1) = segflux(RAY_START, iseg1) - phid;
	    thread_scalar_flux(thread, ireg1, ig) += phid * angle_weights(ray_angle_index(iray), ipol);

            // Backward segment sweep
            ireg2 = ray_fsrs(ray_nsegs(iray) + iseg2 - 1);
            phid = (segflux(RAY_END, iseg2) - source(ireg2, ig)) * exparg(iseg2 - 1);
            segflux(RAY_END, iseg2 - 1) = segflux(RAY_END, iseg2) - phid;
	    thread_scalar_flux(thread, ireg2, ig) += phid * angle_weights(ray_angle_index(iray), ipol);
            iseg2--;
        }
        angflux(ray_bc_index_frwd_end(iray), ipol, ig) = segflux(RAY_START, nsegs);
        angflux(ray_bc_index_bkwd_end(iray), ipol, ig) = segflux(RAY_END, 0);
    });

    for (int k = 0; k < nthreads; k++) {
	for (int i = 0; i < nfsr; i++) {
            for (int j = 0; j < ng; j++) {
		scalar_flux(i, j) += thread_scalar_flux(k, i, j);
	    }
	}
    }

    // Scale the flux with source, volume, and transport XS
    const double fourpi = 4.0 * M_PI;
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = scalar_flux(i, g) * dz / (xstr(i, g) * fsr_vol(i)) + source(i, g) * fourpi;
    });
}

void KokkosMOC::_impl_sweep_serial() {
    using MemSpace = Kokkos::Serial;
    using ExecSpace = MemSpace::execution_space;

    auto& scalar_flux = _h_scalar_flux;
    auto& ray_nsegs = _h_ray_nsegs;
    auto& ray_fsrs = _h_ray_fsrs;
    auto& ray_segments = _h_ray_segments;
    auto& ray_bc_index_frwd_start = _h_ray_bc_index_frwd_start;
    auto& ray_bc_index_frwd_end = _h_ray_bc_index_frwd_end;
    auto& ray_bc_index_bkwd_start = _h_ray_bc_index_bkwd_start;
    auto& ray_bc_index_bkwd_end = _h_ray_bc_index_bkwd_end;
    auto& ray_angle_index = _h_ray_angle_index;
    auto& n_rays = _n_rays;
    auto& npol = _npol;
    auto& nfsr = _nfsr;
    auto& ng = _ng;
    auto& xstr = _h_xstr;
    auto& source = _h_source;
    auto& fsr_vol = _h_fsr_vol;
    auto& dz = _plane_height;
    auto& exparg = _h_exparg;
    auto& segflux = _h_segflux;
    auto& rsinpolang = _h_rsinpolang;
    auto& angle_weights = _h_angle_weights;
    auto& angflux = _h_angflux;
    auto& old_angflux = _h_old_angflux;

    // Initialize the scalar flux to 0.0
    Kokkos::parallel_for("InitializeScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = 0.0;
    });

    // Sweep all the long rays
    Kokkos::parallel_for("SweepRays",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {static_cast<long int>(n_rays), npol, ng}),
        KOKKOS_LAMBDA(int iray, int ipol, int ig) {

            // Store the exponential arguments for this ray
            int nsegs = ray_nsegs(iray + 1) - ray_nsegs(iray);
            for (size_t i = ray_nsegs(iray); i < ray_nsegs(iray + 1); i++) {
                exparg(i - ray_nsegs(iray), ig) = 1.0 - exp(-xstr(ray_fsrs(i), ig) * ray_segments(i) * rsinpolang(ipol));
            }

            // Initialize the ray flux with the angular flux BCs
            segflux(RAY_START, 0, ig) = old_angflux(ray_bc_index_frwd_start(iray), ipol, ig);
            segflux(RAY_END, nsegs, ig) = old_angflux(ray_bc_index_bkwd_start(iray), ipol, ig);

            // Sweep the segments bi-directionally
            int iseg2 = nsegs;
            for (int iseg1 = 0; iseg1 < nsegs; iseg1++) {
                int ireg1 = ray_fsrs(ray_nsegs(iray) + iseg1);
                int ireg2 = ray_fsrs(ray_nsegs(iray) + iseg2 - 1);

                // Forward segment sweep
                double phid = segflux(RAY_START, iseg1, ig) - source(ireg1, ig);
                phid *= exparg(iseg1, ig);
                segflux(RAY_START, iseg1 + 1, ig) = segflux(RAY_START, iseg1, ig) - phid;
                scalar_flux(ireg1, ig) += phid * angle_weights(ray_angle_index(iray), ipol);

                // Backward segment sweep
                phid = segflux(RAY_END, iseg2, ig) - source(ireg2, ig);
                phid *= exparg(iseg2 - 1, ig);
                segflux(RAY_END, iseg2 - 1, ig) = segflux(RAY_END, iseg2, ig) - phid;
                scalar_flux(ireg2, ig) += phid * angle_weights(ray_angle_index(iray), ipol);

                iseg2--;
            }

            // Store the final segments' angular flux into the BCs
            for (size_t ig = 0; ig < ng; ig++) {
                angflux(ray_bc_index_frwd_end(iray), ipol, ig) = segflux(RAY_START, nsegs, ig);
                angflux(ray_bc_index_bkwd_end(iray), ipol, ig) = segflux(RAY_END, 0, ig);
            }
    });

    // Scale the flux with source, volume, and transport XS
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = scalar_flux(i, g) / (xstr(i, g) * fsr_vol(i) / dz) + source(i, g) * 4.0 * M_PI;
    });
}

// Main function to run the serial MOC sweep
void KokkosMOC::sweep() {
    if (_device == "openmp") {
        // Run the MOC sweep using OpenMP
        _impl_sweep_openmp();
    } else if (_device == "serial") {
        // Run the MOC sweep using serial execution
        _impl_sweep_serial();
    }

    Kokkos::deep_copy(_h_old_angflux, _h_angflux);
}
