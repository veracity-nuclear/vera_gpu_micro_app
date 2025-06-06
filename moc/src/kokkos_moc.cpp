#include "kokkos_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include "kokkos_long_ray.hpp"
#include "c5g7_library.hpp"

namespace {
    // Get the total cross sections for each FSR from the library
    Kokkos::View<double**> get_xstr(
        const int num_fsr,
        const int starting_xsr,
        const std::vector<int>& fsr_mat_id,
        const c5g7_library& library
    ) {
        Kokkos::View<double**> xs("xstr", num_fsr, library.get_num_groups());
        for (auto i = 0; i < fsr_mat_id.size(); i++) {
            auto total_xs = library.total(fsr_mat_id[i]);
            for (int g = 0; g < library.get_num_groups(); g++) {
                xs(i, g) = total_xs[g];
            }
        }
        return xs;
    }

    // Reflect the angle for reflecting boundary conditions
    inline int reflect_angle(int angle) {
        return angle % 2 == 0 ? angle + 1 : angle - 1;
    }
}

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
    _fsr_vol = _file.getDataSet("/MOC_Ray_Data/Domain_00001/FSR_Volume").read<std::vector<double>>();
    _plane_height = _file.getDataSet("/MOC_Ray_Data/Domain_00001/plane_height").read<double>();
    _nfsr = _fsr_vol.size();

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
    _xstr = get_xstr(_nfsr, starting_xsr, _fsr_mat_id, _library);

    // Allocate scalar flux and source array
    _scalar_flux = Kokkos::View<double**>("scalar_flux", _nfsr, _ng);
    _source = Kokkos::View<double**>("source", _nfsr, _ng);
    for (size_t i = 0; i < _nfsr; ++i) {
        for (size_t j = 0; j < _ng; ++j) {
            _scalar_flux(i, j) = 1.0;
            _source(i, j) = 1.0;
        }
    }

    // Read ray spacings and angular flux BC dimensions
    auto domain = _file.getGroup("/MOC_Ray_Data/Domain_00001");
    auto polar_angles = _file.getDataSet("/MOC_Ray_Data/Polar_Radians").read<std::vector<double>>();
    auto polar_weights = _file.getDataSet("/MOC_Ray_Data/Polar_Weights").read<std::vector<double>>();
    auto azi_weights = _file.getDataSet("/MOC_Ray_Data/Azimuthal_Weights").read<std::vector<double>>();
    _npol = polar_angles.size();
    int nazi = 0;
    _ray_spacing.clear();
    for (const auto& objName : domain.listObjectNames()) {
        // Loop over each angle group
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            // Read ray spacing
            _ray_spacing.push_back(angleGroup.getDataSet("spacing").read<double>());
            // Read the BC sizes
            int iazi = std::stoi(objName.substr(8)) - 1;
            auto bc_sizes = angleGroup.getDataSet("BC_size").read<std::vector<int>>();
            _angflux.push_back(AngFluxBCAngle(4));
            for (size_t iface = 0; iface < 4; iface++) {
                _angflux[iazi]._faces[iface] = AngFluxBCFace(bc_sizes[iface], _npol, _ng);
            }
            nazi++;
        }
    }
    _old_angflux = _angflux;

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
    _rsinpolang = Kokkos::View<double*>("rsinpolang", _npol);
    for (int ipol = 0; ipol < _npol; ipol++) {
        _rsinpolang(ipol) = 1.0 / std::sin(polar_angles[ipol]);
    }

    // Count maximum segments across all rays
    _max_segments = 0;
    for (int i = 0; i < _rays.size(); i++) {
        _max_segments = std::max(_max_segments, _rays(i)._fsrs.size());
    }

    // Allocate arrays needed during serial sweep
    if (_device == "serial") {
        _segflux = Kokkos::View<double***>("segflux", 2, _max_segments + 1, _ng);
        _exparg = Kokkos::View<double**>("exparg", _max_segments + 1, _ng);
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
    _rays = Kokkos::View<KokkosLongRay*>("rays", nrays);
    nrays = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);

            // Read the radians data from the angle group
            auto radians = angleGroup.getDataSet("Radians").read<double>();
            auto angleIndex = std::stoi(objName.substr(8)) - 1;
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);
                    _rays(nrays) = KokkosLongRay(rayGroup, angleIndex, radians);
                    nrays++;
                }
            }
        }
    }
    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << nrays << " rays from file: " << _filename << std::endl;
}

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
std::vector<double> KokkosMOC::fission_source(const double keff) const {
    std::vector<double> fissrc(_nfsr, 0.0);
    int ixsr = 1;
    for (size_t i = 0; i < _nfsr; i++) {
        if (_library.is_fissile(_fsr_mat_id[i])) {
            for (int g = 0; g < _ng; g++) {
                fissrc[i] += _library.nufiss(_fsr_mat_id[i], g) * _scalar_flux(i, g) / keff;
                // std::cout << i << " " << g << " " << fissrc[i] << " " << library.nufiss(fsr_mat_id[i], g) / keff << " " << scalar_flux[i][g] << std::endl;
            }
        }
    }
    return fissrc;
}

// Build the total source term for each FSR based on the fission source and scattering cross sections
void KokkosMOC::update_source(const std::vector<double>& fissrc) {
    int _nfsr = _scalar_flux.extent(0);
    int ng = _scalar_flux.extent(1);
    int ixsr = 1;
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < ng; g++) {
            _source(i, g) = fissrc[i] * _library.chi(_fsr_mat_id[i], g);
            // std::cout << "mgfs " << i << " " << g << " " << i << " " << fissrc[i] << " " << library.chi(fsr_mat_id[i], g) << " : " << _source[i][g] << std::endl;
            for (int g2 = 0; g2 < ng; g2++) {
                if (g != g2) {
                    _source(i, g) += _library.scat(_fsr_mat_id[i], g, g2) * _scalar_flux(i, g2);
                    // std::cout << "inscatter " << i << " " << g << " " << g2 << " " << " " << _scalar_flux[i][g2] << " " << library.scat(fsr_mat_id[i], g, g2) << " : " << _source[i][g] << std::endl;
                }
            }
            _source(i, g) += _library.self_scat(_fsr_mat_id[i], g) * _scalar_flux(i, g);
            // std::cout << "selfscatter a " << g << " " << i << " " << old_source << " " << library.self_scat(fsr_mat_id[i], g) << " " << _scalar_flux[i][g] << " : " << source[i][g] << std::endl;
            _source(i, g) /= (_library.total(_fsr_mat_id[i], g) * 4.0 * M_PI);
            // std::cout << "selfscatter b " << g << " " << i << " " << 4.0 * M_PI << " " << library.total(fsr_mat_id[i], g) <<  " : " << source[i][g] << std::endl;
        }
    }
}

void KokkosMOC::_impl_sweep_openmp() {
    using MemSpace = Kokkos::OpenMP;
    using ExecSpace = MemSpace::execution_space;

    // Initialize the scalar flux to 0.0
    Kokkos::parallel_for("InitializeScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {_nfsr, _ng}),
        KOKKOS_LAMBDA(int i, int j) {
            _scalar_flux(i, j) = 0.0;
    });

    // Prepare scratch space
    typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
    typedef typename team_policy::member_type team_member;
    team_policy policy(static_cast<long int>(_rays.size()) * _npol * _ng, Kokkos::AUTO, Kokkos::AUTO);
    const size_t bytes_needed_per_team =
        Kokkos::View<double*, typename team_member::scratch_memory_space>::shmem_size(_max_segments + 1) // exparg
        + Kokkos::View<double**, typename team_member::scratch_memory_space>::shmem_size(2, _max_segments + 1); // segflux
    policy.set_scratch_size(0, Kokkos::PerTeam(bytes_needed_per_team));

    // Sweep all the long rays
    Kokkos::parallel_for("Sweep Rays", policy, KOKKOS_LAMBDA(const team_member& teamMember) {
        int iray = teamMember.league_rank() / (_npol * _ng);
        int ipol = (teamMember.league_rank() % (_npol * _ng)) / _ng;
        int ig = teamMember.league_rank() % _ng;
        const auto& ray = _rays(iray);
        Kokkos::View<double*, typename team_member::scratch_memory_space> exparg(teamMember.team_scratch(0), ray._fsrs.size() + 1);
        Kokkos::View<double**, typename team_member::scratch_memory_space> segflux(teamMember.team_scratch(1), 2, ray._fsrs.size() + 1);

        // Allocate and initialize exparg with dimensions [ray._fsrs.size()][_ng]
        for (int j = 0; j < ray._fsrs.size(); j++) {
            exparg(j) = 1.0 - exp(-_xstr(ray._fsrs[j] - 1, ig) * ray._segments[j] * _rsinpolang(ipol));
        }

        segflux(0, 0) =
            ray._bc_index[0] == -1
            ? 0.0
            : _old_angflux[ray.angle()]._faces[ray._bc_face[0]]._angflux[ray._bc_index[0]][ipol][ig];
        segflux(1, ray._fsrs.size()) =
            ray._bc_index[1] == -1
            ? 0.0
            : _old_angflux[ray.angle()]._faces[ray._bc_face[1]]._angflux[ray._bc_index[1]][ipol][ig];

        int iseg2 = ray._fsrs.size();
        double phid;
        for (int iseg1 = 0; iseg1 < ray._fsrs.size(); iseg1++) {
            int ireg1 = ray._fsrs[iseg1] - 1;
            int ireg2 = ray._fsrs[ray._fsrs.size() - iseg1 - 1] - 1;

            // Forward segment sweep
            phid = segflux(0, iseg1) - _source(ireg1, ig);
            phid *= exparg(iseg1);
            segflux(0, iseg1 + 1) = segflux(0, iseg1) - phid;
            Kokkos::atomic_add(&_scalar_flux(ireg1, ig), phid * _angle_weights[ray.angle()][ipol]);

            // Backward segment sweep
            phid = segflux(1, iseg2) - _source(ireg2, ig);
            phid *= exparg(iseg2 - 1);
            segflux(1, iseg2 - 1) = segflux(1, iseg2) - phid;
            Kokkos::atomic_add(&_scalar_flux(ireg2, ig), phid * _angle_weights[ray.angle()][ipol]);
            iseg2--;
        }

        // Store the final segments' angular flux into the BCs
        int refl_angle = reflect_angle(ray.angle());
        if (ray._bc_index[RAY_START] != -1) {
            _angflux[refl_angle]._faces[ray._bc_face[RAY_END]]._angflux[ray._bc_index[RAY_END]][ipol][ig] =
            segflux(RAY_START, ray._fsrs.size());
        }
        if (ray._bc_index[RAY_END] != -1) {
            _angflux[refl_angle]._faces[ray._bc_face[RAY_START]]._angflux[ray._bc_index[RAY_START]][ipol][ig] =
            segflux(RAY_END, 0);
        }
    });

    // Scale the flux with source, volume, and transport XS
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {_nfsr, _ng}),
        KOKKOS_LAMBDA(int i, int g) {
            _scalar_flux(i, g) = _scalar_flux(i, g) / (_xstr(i, g) * _fsr_vol[i] / _plane_height) + _source(i, g) * 4.0 * M_PI;
    });
}

void KokkosMOC::_impl_sweep_serial() {
    using MemSpace = Kokkos::Serial;
    using ExecSpace = MemSpace::execution_space;

    // Initialize the scalar flux to 0.0
    Kokkos::parallel_for("InitializeScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {_nfsr, _ng}),
        KOKKOS_LAMBDA(int i, int g) {
            _scalar_flux(i, g) = 0.0;
    });

    // Sweep all the long rays
    Kokkos::parallel_for("SweepRays",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {static_cast<long int>(_rays.size()), _npol, _ng}),
        KOKKOS_LAMBDA(int i, int ipol, int ig) {

            const auto& ray = _rays(i);

            // Store the exponential arguments for this ray
            for (size_t i = 0; i < ray._fsrs.size(); i++) {
                _exparg(i, ig) = 1.0 - exp(-_xstr( ray._fsrs[i] - 1, ig) * ray._segments[i] * _rsinpolang[ipol]);
            }

            // Initialize the ray flux with the angular flux BCs
            _segflux(0, 0, ig) =
                ray._bc_index[0] == -1
                ? 0.0
                : _old_angflux[ray.angle()]._faces[ray._bc_face[0]]._angflux[ray._bc_index[0]][ipol][ig];
            _segflux(1, ray._fsrs.size(), ig) =
                ray._bc_index[1] == -1
                ? 0.0
                : _old_angflux[ray.angle()]._faces[ray._bc_face[1]]._angflux[ray._bc_index[1]][ipol][ig];

            // Sweep the segments bi-directionally
            int iseg2 = ray._fsrs.size();
            for (int iseg1 = 0; iseg1 < ray._fsrs.size(); iseg1++) {
                int ireg1 = ray._fsrs[iseg1] - 1;
                int ireg2 = ray._fsrs[iseg2 - 1] - 1;

                // Forward segment sweep
                double phid = _segflux(RAY_START, iseg1, ig) - _source(ireg1, ig);
                phid *= _exparg(iseg1, ig);
                _segflux(RAY_START, iseg1 + 1, ig) = _segflux(RAY_START, iseg1, ig) - phid;
                _scalar_flux(ireg1, ig) += phid * _angle_weights[ray.angle()][ipol];

                // Backward segment sweep
                phid = _segflux(RAY_END, iseg2, ig) - _source(ireg2, ig);
                phid *= _exparg(iseg2 - 1, ig);
                _segflux(RAY_END, iseg2 - 1, ig) = _segflux(RAY_END, iseg2, ig) - phid;
                _scalar_flux(ireg2, ig) += phid * _angle_weights[ray.angle()][ipol];

                // throw std::runtime_error("end of first segment");
                iseg2--;
            }

            // Store the final segments' angular flux into the BCs
            int refl_angle = reflect_angle(ray.angle());
            for (size_t ig = 0; ig < _ng; ig++) {
                if (ray._bc_index[RAY_START] != -1) {
                    _angflux[refl_angle]._faces[ray._bc_face[RAY_END]]._angflux[ray._bc_index[RAY_END]][ipol][ig] =
                        _segflux(RAY_START, ray._fsrs.size(), ig);
                }
                if (ray._bc_index[RAY_END] != -1) {
                    _angflux[refl_angle]._faces[ray._bc_face[RAY_START]]._angflux[ray._bc_index[RAY_START]][ipol][ig] =
                        _segflux(RAY_END, 0, ig);
                }
            }
            // throw std::runtime_error("End of segment loop");
    });

    // Scale the flux with source, volume, and transport XS
    Kokkos::parallel_for("ScaleScalarFlux",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {_nfsr, _ng}),
        KOKKOS_LAMBDA(int i, int g) {
            _scalar_flux(i, g) = _scalar_flux(i, g) / (_xstr(i, g) * _fsr_vol[i] / _plane_height) + _source(i, g) * 4.0 * M_PI;
    });

    // Print the scalar flux
    // std::cout << "Scalar Flux:" << std::endl;
    // for (size_t i = 0; i < _scalar_flux.size(); ++i) {
    //     std::cout << "FSR " << i << ": ";
    //     for (size_t g = 0; g < _scalar_flux[i].size(); ++g) {
    //         std::cout << _scalar_flux[i][g] << " ";
    //     }
    //     std::cout << std::endl;
    // }

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

    _old_angflux = _angflux;
}