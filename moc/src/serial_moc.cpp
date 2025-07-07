#include "serial_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include "exp_table.hpp"
#include "c5g7_library.hpp"

SerialMOC::SerialMOC(const ArgumentParser& args) :
    _filename(args.get_positional(0)),
    _file(HighFive::File(_filename, HighFive::File::ReadOnly))
{
    // Read the rays
    _read_rays();

    // Read the FSR volumes and plane height
    _fsr_vol = _file.getDataSet("/MOC_Ray_Data/Domain_00001/FSR_Volume").read<std::vector<double>>();
    _plane_height = _file.getDataSet("/MOC_Ray_Data/Domain_00001/plane_height").read<double>();
    _nfsr = _fsr_vol.size();

    // Get XS
    if (args.get_positional(0) == args.get_positional(1)) {
        _xstr = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xstr").read<std::vector<std::vector<double>>>();
        _xsnf = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xsnf").read<std::vector<std::vector<double>>>();
        _xsch = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xsch").read<std::vector<std::vector<double>>>();
        _xssc = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xssc").read<std::vector<std::vector<std::vector<double>>>>();
	    _ng = _xstr[0].size();
    } else {
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

        auto library = c5g7_library(args.get_positional(1));
        _ng = library.get_num_groups();
        _get_xstr(_nfsr, fsr_mat_id, library);
        _get_xsnf(_nfsr, fsr_mat_id, library);
        _get_xsch(_nfsr, fsr_mat_id, library);
        _get_xssc(_nfsr, fsr_mat_id, library);
    }

    // Allocate scalar flux and source array
    _scalar_flux.resize(_nfsr);
    for (size_t i = 0; i < _nfsr; ++i) {
        _scalar_flux[i].resize(_ng, 1.0);
    }
    _source = _scalar_flux;  // Initialize source to scalar flux

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
            _ray_spacing.push_back(angleGroup.getDataSet("spacing").read<double>());
            // Read the BC sizes
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
        int ang = _ray_angle_index[iray];
        int bc_index = _ray_bc_index_frwd_start[iray];
        if (bc_index >= 0) {
            angface_to_ray[ang][_ray_bc_face_start[iray]][bc_index] = iray;
        }
        bc_index = _ray_bc_index_bkwd_start[iray];
        if (bc_index >= 0) {
            angface_to_ray[ang][_ray_bc_face_end[iray]][bc_index] = _n_rays + iray;
        }
    }

    // Now allocate the angular flux arrays, remap the long ray indexes, and initialize the angular flux arrays
    total_bc_points = 2 * total_bc_points + 2;  // Both directions on each ray, plus two for the vacuum rays
    _angflux.resize(total_bc_points);
    for (size_t i = 0; i < total_bc_points; i++) {
        _angflux[i].resize(_npol);
        for (size_t j = 0; j < _npol; j++) {
            _angflux[i][j].resize(_ng);
        }
    }
    _old_angflux = _angflux;
    for (size_t i = 0; i < _n_rays; i++) {
        int ang = _ray_angle_index[i];
        int irefl = ang % 2 == 0 ? ang + 1 : ang - 1;
        if (_ray_bc_index_frwd_start[i] == -1) {
            _ray_bc_index_frwd_start[i] = total_bc_points - 2;
            _ray_bc_index_bkwd_end[i] = total_bc_points - 1;
        } else {
            int start_index = _ray_bc_index_frwd_start[i];
            _ray_bc_index_frwd_start[i] = angface_to_ray[ang][_ray_bc_face_start[i]][start_index];
            _ray_bc_index_bkwd_end[i] = angface_to_ray[irefl][_ray_bc_face_start[i]][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _angflux[_ray_bc_index_frwd_start[i]][ipol][ig] = 0.0;
                    _angflux[_ray_bc_index_bkwd_end[i]][ipol][ig] = 0.0;
                }
            }
        }
        if (_ray_bc_index_frwd_end[i] == -1) {
            _ray_bc_index_bkwd_start[i] = total_bc_points - 2;
            _ray_bc_index_frwd_end[i] = total_bc_points - 1;
        } else {
            int start_index = _ray_bc_index_bkwd_start[i];
            _ray_bc_index_frwd_end[i] = angface_to_ray[irefl][_ray_bc_face_end[i]][start_index];
            _ray_bc_index_bkwd_start[i] = angface_to_ray[ang][_ray_bc_face_end[i]][start_index];
            for (size_t ipol = 0; ipol < _npol; ipol++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _angflux[_ray_bc_index_frwd_end[i]][ipol][ig] = 0.0;
                    _angflux[_ray_bc_index_bkwd_start[i]][ipol][ig] = 0.0;
                }
            }
        }
    }
    _old_angflux = _angflux;  // Initialize old angular flux to current angular flux

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
    _rsinpolang.resize(_npol);
    for (int ipol = 0; ipol < _npol; ipol++) {
        _rsinpolang[ipol] = 1.0 / std::sin(polar_angles[ipol]);
    }

    // Allocate segment flux array
    _max_segments = 0;
    for (int i = 0; i < _n_rays; i++) {
        _max_segments = std::max<int>(_max_segments, _ray_nsegs[i + 1] - _ray_nsegs[i]);
    }

    // Initialize the exponential argument array
    _exparg.resize(_max_segments + 1, 0.0);
}

void SerialMOC::_read_rays() {
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
    _ray_nsegs.resize(_n_rays + 1);
    _ray_bc_face_start.resize(_n_rays);
    _ray_bc_face_end.resize(_n_rays);
    _ray_bc_index_frwd_start.resize(_n_rays);
    _ray_bc_index_frwd_end.resize(_n_rays);
    _ray_bc_index_bkwd_start.resize(_n_rays);
    _ray_bc_index_bkwd_end.resize(_n_rays);
    _ray_angle_index.resize(_n_rays);
    _ray_nsegs[0] = 0;

    int iray = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            auto angleIndex = std::stoi(objName.substr(8)) - 1;
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);

                    // Read ray data
                    auto fsrs = rayGroup.getDataSet("FSRs").read<std::vector<int>>();
                    auto segments = rayGroup.getDataSet("Segments").read<std::vector<double>>();
                    auto bc_face = rayGroup.getDataSet("BC_face").read<std::vector<int>>();
                    auto bc_index = rayGroup.getDataSet("BC_index").read<std::vector<int>>();

                    // Store ray metadata
                    _ray_angle_index[iray] = angleIndex;
                    _ray_nsegs[iray + 1] = _ray_nsegs[iray] + fsrs.size();

                    // Adjust BC data (convert from 1-based to 0-based indexing)
                    _ray_bc_face_start[iray] = bc_face[RAY_START] - 1;
                    _ray_bc_face_end[iray] = bc_face[RAY_END] - 1;
                    _ray_bc_index_frwd_start[iray] = bc_index[RAY_START] - 1;
                    _ray_bc_index_frwd_end[iray] = bc_index[RAY_END] - 1;
                    _ray_bc_index_bkwd_start[iray] = bc_index[RAY_END] - 1;
                    _ray_bc_index_bkwd_end[iray] = bc_index[RAY_START] - 1;

                    iray++;
                }
            }
        }
    }

    // Allocate flattened segment data
    _ray_fsrs.resize(_ray_nsegs[_n_rays]);
    _ray_segments.resize(_ray_nsegs[_n_rays]);

    // Fill flattened segment data
    iray = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);

                    // Read segment data
                    auto fsrs = rayGroup.getDataSet("FSRs").read<std::vector<int>>();
                    auto segments = rayGroup.getDataSet("Segments").read<std::vector<double>>();

                    // Store segment data in flattened arrays
                    for (int iseg = 0; iseg < fsrs.size(); iseg++) {
                        int global_seg = _ray_nsegs[iray] + iseg;
                        _ray_fsrs[global_seg] = fsrs[iseg];
                        _ray_segments[global_seg] = segments[iseg];
                    }

                    iray++;
                }
            }
        }
    }

    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << _n_rays << " rays from file: " << _filename << std::endl;
}

// Get the total cross sections for each FSR from the library
void SerialMOC::_get_xstr(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _xstr.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        _xstr[i] = library.total(fsr_mat_id[i]);
    }
}

// Get the nu-fission cross sections for each FSR from the library
void SerialMOC::_get_xsnf(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _xsnf.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        _xsnf[i] = library.nufiss(fsr_mat_id[i]);
    }
}

// Get the chi for each FSR from the library
void SerialMOC::_get_xsch(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _xsch.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        _xsch[i] = library.chi(fsr_mat_id[i]);
    }
}

// Get the scattering cross sections for each FSR from the library
void SerialMOC::_get_xssc(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _xssc.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        _xssc[i].resize(_ng);
        for (int g = 0; g < _ng; g++) {
            _xssc[i][g].resize(_ng);
            for (int g2 = 0; g2 < _ng; g2++) {
                _xssc[i][g][g2] = library.scat(fsr_mat_id[i], g, g2);
            }
        }
    }
}

// Reflect the angle for reflecting boundary conditions
inline int reflect_angle(int angle) {
    return angle % 2 == 0 ? angle + 1 : angle - 1;
}

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
std::vector<double> SerialMOC::fission_source(const double keff) const {
    std::vector<double> fissrc(_nfsr, 0.0);
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < _ng; g++) {
            fissrc[i] += _xsnf[i][g] * _scalar_flux[i][g] / keff;
        }
    }
    return fissrc;
}

// Build the total source term for each FSR based on the fission source and scattering cross sections
void SerialMOC::update_source(const std::vector<double>& fissrc) {
    int _nfsr = _scalar_flux.size();
    int ng = _scalar_flux[0].size();
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < ng; g++) {
            _source[i][g] = fissrc[i] * _xsch[i][g];
            for (int g2 = 0; g2 < ng; g2++) {
                if (g != g2) {
                    _source[i][g] += _xssc[i][g][g2] * _scalar_flux[i][g2];
                }
            }
            _source[i][g] += _xssc[i][g][g] * _scalar_flux[i][g];
            _source[i][g] /= (_xstr[i][g] * 4.0 * M_PI);
        }
    }
}

// Main function to run the serial MOC sweep
void SerialMOC::sweep() {
    Kokkos::Profiling::pushRegion("SerialMOC::Sweep");
    // Initialize old values and a few scratch values
    int iseg1, iseg2, ireg, refl_angle, global_seg;
    double phid1, phid2;

    // Initialize the scalar flux to 0.0
    for (auto i = 0; i < _nfsr; i++) {
        for (auto g = 0; g < _ng; g++) {
            _scalar_flux[i][g] = 0.0;
        }
    }

    // Sweep all the long rays
    for (int iray = 0; iray < _n_rays; iray++) {
        int nsegs = _ray_nsegs[iray + 1] - _ray_nsegs[iray];

        // Sweep all the polar angles
        for (size_t ipol = 0; ipol < _npol; ipol++) {
            for (size_t ig = 0; ig < _ng; ig++) {

                // Store the exponential arguments for this ray
                for (size_t i = 0; i < nsegs; i++) {
                    int global_seg = _ray_nsegs[iray] + i;
                    _exparg[i] = _exp_table.expt(-_xstr[_ray_fsrs[global_seg] - 1][ig] * _ray_segments[global_seg] * _rsinpolang[ipol]);
                }

                double fsegflux = _old_angflux[_ray_bc_index_frwd_start[iray]][ipol][ig];
                double bsegflux = _old_angflux[_ray_bc_index_bkwd_start[iray]][ipol][ig];

                // Forward sweep
                iseg2 = nsegs;
                for (int iseg1 = 0; iseg1 < nsegs; iseg1++) {
                    global_seg = _ray_nsegs[iray] + iseg1;
                    ireg = _ray_fsrs[global_seg] - 1;
                    phid1 = fsegflux - _source[ireg][ig];
                    phid1 *= _exparg[iseg1];
                    fsegflux = fsegflux - phid1;
                    _scalar_flux[ireg][ig] += phid1 * _angle_weights[_ray_angle_index[iray]][ipol];

                    int global_seg = _ray_nsegs[iray] + iseg2 - 1;
                    ireg = _ray_fsrs[global_seg] - 1;
                    phid2 = bsegflux - _source[ireg][ig];
                    phid2 *= _exparg[iseg2 - 1];
                    bsegflux = bsegflux - phid2;
                    _scalar_flux[ireg][ig] += phid2 * _angle_weights[_ray_angle_index[iray]][ipol];
                    iseg2--;
                }
                _angflux[_ray_bc_index_frwd_end[iray]][ipol][ig] = fsegflux;
                _angflux[_ray_bc_index_bkwd_end[iray]][ipol][ig] = bsegflux;
            }
        }
    }

    // Scale the flux with source, volume, and transport XS
    for (size_t i = 0; i < _nfsr; ++i) {
        for (size_t g = 0; g < _ng; ++g) {
            _scalar_flux[i][g] = _scalar_flux[i][g] / (_xstr[i][g] * _fsr_vol[i] / _plane_height) + _source[i][g] * 4.0 * M_PI;
        }
    }

    _old_angflux = _angflux;
    Kokkos::Profiling::popRegion();
}
