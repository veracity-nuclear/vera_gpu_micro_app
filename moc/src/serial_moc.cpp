#include "serial_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include "exp_table.hpp"
#include "long_ray.hpp"
#include "c5g7_library.hpp"

// Get the total cross sections for each FSR from the library
std::vector<std::vector<double>> get_xstr(
    const int num_fsr,
    const int starting_xsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    std::vector<std::vector<double>> xs;
    xs.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        xs[i] = library.total(fsr_mat_id[i]);
    }
    return xs;
}

SerialMOC::SerialMOC(const std::string& filename, const std::string& libname)
    : _filename(filename), _library(c5g7_library(libname)), _file(HighFive::File(filename, HighFive::File::ReadOnly)) {

    // Process the file here
    _file = HighFive::File(filename, HighFive::File::ReadOnly);

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
    _rsinpolang.resize(_npol);
    for (int ipol = 0; ipol < _npol; ipol++) {
        _rsinpolang[ipol] = 1.0 / std::sin(polar_angles[ipol]);
    }

    // Allocate segment flux array
    _max_segments = 0;
    for (const auto& ray : _rays) {
        _max_segments = std::max(_max_segments, ray._fsrs.size());
    }
    _segflux.resize(2);
    for (size_t j = 0; j < 2; j++) {
        _segflux[j].resize(_max_segments + 1);
        for (size_t i = 0; i < _max_segments + 1; i++) {
            _segflux[j][i].resize(_ng, 0.0);
        }
    }

    // Initialize the exponential argument array
    _exparg.resize(_max_segments + 1);
    for (size_t i = 0; i < _max_segments + 1; i++) {
        _exparg[i].resize(_ng, 0.0);
    }
}

void SerialMOC::_read_rays() {
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
    _rays.reserve(nrays);
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
                    _rays.push_back(LongRay(rayGroup, angleIndex, radians));
                    nrays++;
                }
            }
        }
    }
    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << nrays << " rays from file: " << _filename << std::endl;
}

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
std::vector<double> SerialMOC::fission_source(const double keff) const {
    std::vector<double> fissrc(_nfsr, 0.0);
    int ixsr = 1;
    for (size_t i = 0; i < _nfsr; i++) {
        if (_library.is_fissile(_fsr_mat_id[i])) {
            for (int g = 0; g < _ng; g++) {
                fissrc[i] += _library.nufiss(_fsr_mat_id[i], g) * _scalar_flux[i][g] / keff;
                // std::cout << i << " " << g << " " << fissrc[i] << " " << library.nufiss(fsr_mat_id[i], g) / keff << " " << scalar_flux[i][g] << std::endl;
            }
        }
    }
    return fissrc;
}

// Build the total source term for each FSR based on the fission source and scattering cross sections
void SerialMOC::update_source(const std::vector<double>& fissrc) {
    int _nfsr = _scalar_flux.size();
    int ng = _scalar_flux[0].size();
    int ixsr = 1;
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < ng; g++) {
            _source[i][g] = fissrc[i] * _library.chi(_fsr_mat_id[i], g);
            // std::cout << "mgfs " << i << " " << g << " " << i << " " << fissrc[i] << " " << library.chi(fsr_mat_id[i], g) << " : " << _source[i][g] << std::endl;
            for (int g2 = 0; g2 < ng; g2++) {
                if (g != g2) {
                    _source[i][g] += _library.scat(_fsr_mat_id[i], g, g2) * _scalar_flux[i][g2];
                    // std::cout << "inscatter " << i << " " << g << " " << g2 << " " << " " << _scalar_flux[i][g2] << " " << library.scat(fsr_mat_id[i], g, g2) << " : " << _source[i][g] << std::endl;
                }
            }
            _source[i][g] += _library.self_scat(_fsr_mat_id[i], g) * _scalar_flux[i][g];
            // std::cout << "selfscatter a " << g << " " << i << " " << old_source << " " << library.self_scat(fsr_mat_id[i], g) << " " << _scalar_flux[i][g] << " : " << source[i][g] << std::endl;
            _source[i][g] /= (_library.total(_fsr_mat_id[i], g) * 4.0 * M_PI);
            // std::cout << "selfscatter b " << g << " " << i << " " << 4.0 * M_PI << " " << library.total(fsr_mat_id[i], g) <<  " : " << source[i][g] << std::endl;
        }
    }
}

// Reflect the angle for reflecting boundary conditions
int reflect_angle(int angle) {
    return angle % 2 == 0 ? angle + 1 : angle - 1;
}

// Main function to run the serial MOC sweep
void SerialMOC::sweep() {
    // Initialize old values and a few scratch values
    int iseg1, iseg2, ireg1, ireg2, refl_angle;
    double phid1, phid2, phio1, phio2;

    // Initialize the scalar flux to 0.0
    for (auto i = 0; i < _nfsr; i++) {
        for (auto g = 0; g < _ng; g++) {
            _scalar_flux[i][g] = 0.0;
            // std::cout << "source " << i << " " << g << " " << source[i][g] << " " << _old__scalar_flux[i][g] << std::endl;
        }
    }

    // Sweep all the long rayse
    for (const auto& ray : _rays) {
        // if (ray.angle() == 2) {
        //     throw std::runtime_error("Beginning of ray loop");
        // }

        // Sweep all the polar angles
        for (size_t ipol = 0; ipol < _npol; ipol++) {

            // Store the exponential arguments for this ray
            for (size_t i = 0; i < ray._fsrs.size(); i++) {
                for (size_t ig = 0; ig < _ng; ig++) {
                    _exparg[i][ig] = _exp_table.expt(-_xstr[ray._fsrs[i] - 1][ig] * ray._segments[i] * _rsinpolang[ipol]);
                }
            }

            // Initialize the ray flux with the angular flux BCs
            for (size_t ig = 0; ig < _ng; ig++) {
                _segflux[0][0][ig] =
                    ray._bc_index[0] == -1
                    ? 0.0
                    : _old_angflux[ray.angle()]._faces[ray._bc_face[0]]._angflux[ray._bc_index[0]][ipol][ig];
                _segflux[1][ray._fsrs.size()][ig] =
                    ray._bc_index[1] == -1
                    ? 0.0
                    : _old_angflux[ray.angle()]._faces[ray._bc_face[1]]._angflux[ray._bc_index[1]][ipol][ig];
            }

            // Sweep the segments bi-directionally
            iseg2 = ray._fsrs.size();
            for (iseg1 = 0; iseg1 < ray._fsrs.size(); iseg1++) {
                ireg1 = ray._fsrs[iseg1] - 1;
                ireg2 = ray._fsrs[iseg2 - 1] - 1;

                // Sweep the groups on the 2 segments
                for (size_t ig = 0; ig < _ng; ig++) {
                    // Forward segment sweep
                    phid1 = _segflux[RAY_START][iseg1][ig] - _source[ireg1][ig];
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " " << _segflux[RAY_START][iseg1][ig] << " " << source[ireg1][ig] << " " << phid1 << std::endl;
                    // }
                    // TODO: tabulate exp
                    phid1 *= _exparg[iseg1][ig];
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                    //         << 1.0 - std::exp(-_xstr[ireg1][ig] * ray._segments[iseg1] * _rsinpolang[ipol]) << " : " << phid1 << std::endl;
                    // }
                    _segflux[RAY_START][iseg1 + 1][ig] = _segflux[RAY_START][iseg1][ig] - phid1;
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                    //         << _segflux[RAY_START][iseg1 + 1][ig] << " " << _segflux[RAY_START][iseg1][ig] << " " << phid1
                    //         << std::endl;
                    // }
                    _scalar_flux[ireg1][ig] += phid1 * _angle_weights[ray.angle()][ipol];
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                    //         << _scalar_flux[ireg1][ig] << " " << phid1 << " " << _angle_weights[ray.angle()][ipol] << std::endl;
                    // }

                    // Backward segment sweep
                    phid2 = _segflux[RAY_END][iseg2][ig] - _source[ireg2][ig];
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " " << _segflux[RAY_END][iseg2][ig] << " " << source[ireg2][ig] << " " << phid2 << std::endl;
                    // }
                    phid2 *= _exparg[iseg2 - 1][ig];
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " "
                    //     << 1.0 - std::exp(-_xstr[ireg2][ig] * ray._segments[iseg2 - 1] * _rsinpolang[ipol]) << " "
                    //     << " : " << phid2 << std::endl;
                    // }
                    _segflux[RAY_END][iseg2 - 1][ig] = _segflux[RAY_END][iseg2][ig] - phid2;
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " "
                    //         << _segflux[RAY_END][iseg2 - 1][ig] << " " << _segflux[RAY_END][iseg2][ig] << " " << phid2 << std::endl;
                    // }
                    _scalar_flux[ireg2][ig] += phid2 * _angle_weights[ray.angle()][ipol];
                    // if (ray.angle() == debug_angle) {
                    //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " "
                    //         << _scalar_flux[ireg2][ig] << " " << phid2 << " " << _angle_weights[ray.angle()][ipol] << std::endl;
                    // }
                }
                // throw std::runtime_error("end of first segment");
                iseg2--;
            }

            // Store the final segments' angular flux into the BCs
            for (size_t ig = 0; ig < _ng; ig++) {
                refl_angle = reflect_angle(ray.angle());
                if (ray._bc_index[RAY_START] != -1) {
                    _angflux[refl_angle]._faces[ray._bc_face[RAY_END]]._angflux[ray._bc_index[RAY_END]][ipol][ig] =
                        _segflux[RAY_START][iseg1][ig];
                }
                refl_angle = reflect_angle(ray.angle());
                if (ray._bc_index[RAY_END] != -1) {
                    _angflux[refl_angle]._faces[ray._bc_face[RAY_START]]._angflux[ray._bc_index[RAY_START]][ipol][ig] =
                        _segflux[RAY_END][0][ig];
                }
            }
            // throw std::runtime_error("End of segment loop");
        }
        // throw std::runtime_error("End of polar loop");
    }

    // Scale the flux with source, volume, and transport XS
    for (size_t i = 0; i < _nfsr; ++i) {
        for (size_t g = 0; g < _ng; ++g) {
            // std::cout << i << " " << g << " scale " << _scalar_flux[i][g] << " " << _xstr[i][g] << " " << _fsr_vol[i] << " " << _plane_height << " " << source[i][g] << " " << 4.0 * M_PI << std::endl;
            _scalar_flux[i][g] = _scalar_flux[i][g] / (_xstr[i][g] * _fsr_vol[i] / _plane_height) + _source[i][g] * 4.0 * M_PI;
        }
    }

    // Print the scalar flux
    // std::cout << "Scalar Flux:" << std::endl;
    // for (size_t i = 0; i < _scalar_flux.size(); ++i) {
    //     std::cout << "FSR " << i << ": ";
    //     for (size_t g = 0; g < _scalar_flux[i].size(); ++g) {
    //         std::cout << _scalar_flux[i][g] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    _old_angflux = _angflux;
}