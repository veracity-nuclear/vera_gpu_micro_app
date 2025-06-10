#include "kokkos_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <cassert>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include "kokkos_long_ray.hpp"
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
    for (size_t iray = 0; iray < _h_rays.size(); iray++) {
        auto& ray = _h_rays(iray);
        if (ray._bc_index_frwd_start >= 0) {
            angface_to_ray[ray.angle()][ray._bc_face_start][ray._bc_index_frwd_start] = iray;
        }
        if (ray._bc_index_bkwd_start >= 0) {
            angface_to_ray[ray.angle()][ray._bc_face_end][ray._bc_index_bkwd_start] = _h_rays.size() + iray;
        }
    }

    // Now allocate the angular flux arrays, remap the long ray indexes, and initialize the angular flux arrays
    total_bc_points = 2 * total_bc_points + 2;  // Both directions on each ray, plus two for the vacuum rays
    _h_angflux = Kokkos::View<double***, Kokkos::HostSpace>("angflux", total_bc_points, _npol, _ng);
    _h_old_angflux = Kokkos::View<double***, Kokkos::HostSpace>("old_angflux", total_bc_points, _npol, _ng);
    for (size_t i = 0; i < _h_rays.size(); i++) {
        auto& ray = _h_rays(i);
        int irefl = ray.angle() % 2 == 0 ? ray.angle() + 1 : ray.angle() - 1;
        if (ray._bc_index_frwd_start == -1) {
            ray._bc_index_frwd_start = total_bc_points - 2;
            ray._bc_index_bkwd_end = total_bc_points - 1;
        } else {
            int start_index = ray._bc_index_frwd_start;
            ray._bc_index_frwd_start = angface_to_ray[ray.angle()][ray._bc_face_start][start_index];
            ray._bc_index_bkwd_end = angface_to_ray[irefl][ray._bc_face_start][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _h_angflux(ray._bc_index_frwd_start, ipol, ig) = 0.0;
                    _h_angflux(ray._bc_index_bkwd_end, ipol, ig) = 0.0;
                }
            }
        }
        if (ray._bc_index_frwd_end == -1) {
            ray._bc_index_bkwd_start = total_bc_points - 2;
            ray._bc_index_frwd_end = total_bc_points - 1;
        } else {
            int start_index = ray._bc_index_bkwd_start;
            ray._bc_index_frwd_end = angface_to_ray[irefl][ray._bc_face_end][start_index];
            ray._bc_index_bkwd_start = angface_to_ray[ray.angle()][ray._bc_face_end][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _h_angflux(ray._bc_index_frwd_end, ipol, ig) = 0.0;
                    _h_angflux(ray._bc_index_bkwd_start, ipol, ig) = 0.0;
                }
            }
        }
    }
    Kokkos::deep_copy(_h_old_angflux, _h_angflux);

    // Build angle weights
    _angle_weights.reserve(nazi);
    for (int iazi = 0; iazi < nazi; iazi++) {
        _angle_weights.push_back(std::vector<double>(_npol, 0.0));
        for (int ipol = 0; ipol < _angle_weights[iazi].size(); ipol++) {
            _angle_weights[iazi][ipol] = _ray_spacing[iazi] * azi_weights[iazi] * polar_weights[ipol]
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
    for (int i = 0; i < _h_rays.size(); i++) {
        _max_segments = std::max(_max_segments, _h_rays(i)._fsrs.size());
    }

    // Allocate arrays needed during serial sweep
    if (_device == "serial") {
        _h_segflux = Kokkos::View<double***, Kokkos::HostSpace>("segflux", 2, _max_segments + 1, _ng);
        _h_exparg = Kokkos::View<double**, Kokkos::HostSpace>("exparg", _max_segments + 1, _ng);
    }

    // Set up device views as needed
    if (_device == "cuda") {
        _d_fsr_vol = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), _h_fsr_vol);
        _d_rays = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), _h_rays);
        _d_rsinpolang = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(), _h_rsinpolang);
	Kokkos::deep_copy(_d_xstr, _h_xstr);
	Kokkos::deep_copy(_d_scalar_flux, _h_scalar_flux);
	Kokkos::deep_copy(_d_source, _h_source);
    }
}

void KokkosMOC::_read_rays() {
    auto domain = _file.getGroup("/MOC_Ray_Data/Domain_00001");

    // Count the rays
    auto nrays = 0;
    auto nangles = 0;
    for (size_t i = 0; i < domain.listObjectNames().size(); i++) {
        std::string objName = domain.listObjectNames()[i];
        if (objName.substr(0, 6) == "Angle_") {
            auto angleGroup = domain.getGroup(objName);
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    nrays++;
                }
            }
            nangles++;
        }
    }

    // Set up the rays
    _h_rays = Kokkos::View<KokkosLongRay*, Kokkos::HostSpace>("rays", nrays);
    nrays = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);

            // Read the data from the angle group
            auto angleIndex = std::stoi(objName.substr(8)) - 1;
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);
                    _h_rays(nrays) = KokkosLongRay(rayGroup, angleIndex);
                    nrays++;
                }
            }
        }
    }
    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << nrays << " rays from file: " << _filename << std::endl;
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
    auto& rays = _h_rays;
    auto npol = _npol;
    auto nfsr = _nfsr;
    auto ng = _ng;

    // Initialize the scalar flux to 0.0
    Kokkos::parallel_for("InitializeScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int j) {
            scalar_flux(i, j) = 0.0;
    });

    // Prepare scratch space
    typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
    typedef typename team_policy::member_type team_member;
    team_policy policy(static_cast<long int>(rays.size()) * npol * ng, Kokkos::AUTO, Kokkos::AUTO);
    const size_t bytes_needed_per_team =
        Kokkos::View<double*, typename team_member::scratch_memory_space>::shmem_size(_max_segments + 1) // exparg
        + Kokkos::View<double*, typename team_member::scratch_memory_space>::shmem_size(_max_segments + 1); // segflux
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes_needed_per_team));

    // Sweep all the long rays
    Kokkos::parallel_for("Sweep Rays", policy, KOKKOS_LAMBDA(const team_member& teamMember) {
        int iray = teamMember.league_rank() / (npol * ng);
        int ipol = (teamMember.league_rank() % (npol * ng)) / ng;
        int ig = teamMember.league_rank() % ng;
        const auto& ray = rays(iray);
        Kokkos::View<double*, typename team_member::scratch_memory_space> exparg(teamMember.team_scratch(0), ray._fsrs.size() + 1);
        Kokkos::View<double*, typename team_member::scratch_memory_space> segflux(teamMember.team_scratch(0), ray._fsrs.size() + 1);

        // Allocate and initialize exparg with dimensions [ray._fsrs.size()][ng]
        for (int j = 0; j < ray._fsrs.size(); j++) {
            exparg(j) = 1.0 - exp(-_h_xstr(ray._fsrs[j] - 1, ig) * ray._segments[j] * _h_rsinpolang(ipol));
        }

        int ireg;
        double phid;

        // Forward segment sweep
        segflux(0) = _h_old_angflux(ray._bc_index_frwd_start, ipol, ig);
        for (int iseg = 0; iseg < ray._fsrs.size(); iseg++) {
            ireg = ray._fsrs[iseg] - 1;
            phid = (segflux(iseg) - _h_source(ireg, ig)) * exparg(iseg);
            segflux(iseg + 1) = segflux(iseg) - phid;
            Kokkos::atomic_add(&scalar_flux(ireg, ig), phid * _angle_weights[ray.angle()][ipol]);
        }
        _h_angflux(ray._bc_index_frwd_end, ipol, ig) = segflux(ray._fsrs.size());

        // Backward segment sweep
        segflux(ray._fsrs.size()) = _h_old_angflux(ray._bc_index_bkwd_start, ipol, ig);
        for (int iseg = ray._fsrs.size(); iseg > 0; iseg--) {
            ireg = ray._fsrs[iseg - 1] - 1;
            phid = (segflux(iseg) - _h_source(ireg, ig)) * exparg(iseg - 1);
            segflux(iseg - 1) = segflux(iseg) - phid;
            Kokkos::atomic_add(&scalar_flux(ireg, ig), phid * _angle_weights[ray.angle()][ipol]);
        }
        _h_angflux(ray._bc_index_bkwd_end, ipol, ig) = segflux(0);
    });

    // Scale the flux with source, volume, and transport XS
    const double fourpi = 4.0 * M_PI;
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = scalar_flux(i, g) * _plane_height / (_h_xstr(i, g) * _h_fsr_vol(i)) + _h_source(i, g) * fourpi;
    });
}

void KokkosMOC::_impl_sweep_serial() {
    using MemSpace = Kokkos::Serial;
    using ExecSpace = MemSpace::execution_space;

    auto& scalar_flux = _h_scalar_flux;
    auto& rays = _h_rays;
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
    auto& angle_weights = _angle_weights;

    // Initialize the scalar flux to 0.0
    Kokkos::parallel_for("InitializeScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nfsr, ng}),
        KOKKOS_LAMBDA(int i, int g) {
            scalar_flux(i, g) = 0.0;
    });

    // Sweep all the long rays
    Kokkos::parallel_for("SweepRays",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {static_cast<long int>(rays.size()), npol, ng}),
        KOKKOS_LAMBDA(int i, int ipol, int ig) {

            const auto& ray = rays(i);

            // Store the exponential arguments for this ray
            for (size_t i = 0; i < ray._fsrs.size(); i++) {
                exparg(i, ig) = 1.0 - exp(-xstr(ray._fsrs[i] - 1, ig) * ray._segments[i] * rsinpolang(ipol));
            }

            // Initialize the ray flux with the angular flux BCs
            segflux(RAY_START, 0, ig) = _h_old_angflux(ray._bc_index_frwd_start, ipol, ig);
            segflux(RAY_END, ray._fsrs.size(), ig) = _h_old_angflux(ray._bc_index_bkwd_start, ipol, ig);

            // Sweep the segments bi-directionally
            int iseg2 = ray._fsrs.size();
            for (int iseg1 = 0; iseg1 < ray._fsrs.size(); iseg1++) {
                int ireg1 = ray._fsrs[iseg1] - 1;
                int ireg2 = ray._fsrs[iseg2 - 1] - 1;

                // Forward segment sweep
                double phid = segflux(RAY_START, iseg1, ig) - source(ireg1, ig);
                phid *= exparg(iseg1, ig);
                segflux(RAY_START, iseg1 + 1, ig) = segflux(RAY_START, iseg1, ig) - phid;
                scalar_flux(ireg1, ig) += phid * angle_weights[ray.angle()][ipol];

                // Backward segment sweep
                phid = segflux(RAY_END, iseg2, ig) - source(ireg2, ig);
                phid *= exparg(iseg2 - 1, ig);
                segflux(RAY_END, iseg2 - 1, ig) = segflux(RAY_END, iseg2, ig) - phid;
                scalar_flux(ireg2, ig) += phid * angle_weights[ray.angle()][ipol];

                iseg2--;
            }

            // Store the final segments' angular flux into the BCs
            for (size_t ig = 0; ig < ng; ig++) {
                _h_angflux(ray._bc_index_frwd_end, ipol, ig) = segflux(RAY_START, ray._fsrs.size(), ig);
                _h_angflux(ray._bc_index_bkwd_end, ipol, ig) = segflux(RAY_END, 0, ig);
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
