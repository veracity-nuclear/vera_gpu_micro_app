#include "serial_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include "exp_table.hpp"
#include "long_ray.hpp"
#include "c5g7_library.hpp"

template <typename RealType>
SerialMOC<RealType>::SerialMOC(const ArgumentParser& args) :
    _filename(args.get_positional(0)),
    _file(HighFive::File(_filename, HighFive::File::ReadOnly)),
    _ray_sort(args.get_option("ray_sort"))
{
    // Read the rays
    _read_rays();

    // Read the FSR volumes and plane height
    auto fsr_vol_tmp = _file.getDataSet("/MOC_Ray_Data/Domain_00001/FSR_Volume").read<std::vector<double>>();
    _fsr_vol.resize(fsr_vol_tmp.size());
    for (size_t i = 0; i < fsr_vol_tmp.size(); i++) {
        _fsr_vol[i] = static_cast<RealType>(fsr_vol_tmp[i]);
    }
    _plane_height = static_cast<RealType>(_file.getDataSet("/MOC_Ray_Data/Domain_00001/plane_height").read<double>());
    _nfsr = _fsr_vol.size();

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
        auto xstr = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xstr").read<std::vector<std::vector<double>>>();
        auto xsnf = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xsnf").read<std::vector<std::vector<double>>>();
        auto xsch = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xsch").read<std::vector<std::vector<double>>>();
        auto xssc = _file.getDataSet("MOC_Ray_Data/Domain_00001/Solution_Data/xssc").read<std::vector<std::vector<std::vector<double>>>>();
        _ng = xstr[0].size();
        _xstr.resize(_nfsr);
        _xsnf.resize(_nfsr);
        _xsch.resize(_nfsr);
        _xssc.resize(_nfsr);
        int ixsr = 0;
        for (int i = 0; i < _nfsr; i++) {
            if (i == xsrToFsrMap[ixsr]) {
                ixsr++;
            }
            _xstr[i].resize(_ng);
            _xsnf[i].resize(_ng);
            _xsch[i].resize(_ng);
            _xssc[i].resize(_ng);
            for (int to = 0; to < _ng; to++) {
                _xstr[i][to] = static_cast<RealType>(xstr[ixsr - 1][to]);
                _xsnf[i][to] = static_cast<RealType>(xsnf[ixsr - 1][to]);
                _xsch[i][to] = static_cast<RealType>(xsch[ixsr - 1][to]);
                _xssc[i][to].resize(_ng);
                for (int from = 0; from < _ng; from++) {
                    _xssc[i][to][from] = static_cast<RealType>(xssc[ixsr - 1][to][from]);
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
    _scalar_flux.resize(_nfsr);
    _source.resize(_nfsr);
    for (size_t i = 0; i < _nfsr; ++i) {
        _scalar_flux[i].resize(_ng, 1.0);
        _source[i].resize(_ng, static_cast<RealType>(1.0));
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
            _ray_spacing.push_back(static_cast<RealType>(angleGroup.getDataSet("spacing").read<double>()));
            // Read the BC sizes
            int iazi = std::stoi(objName.substr(8)) - 1;
            auto bc_sizes = angleGroup.getDataSet("BC_size").read<std::vector<int>>();
            _angflux.push_back(AngFluxBCAngle<RealType>(4));
            for (size_t iface = 0; iface < 4; iface++) {
                _angflux[iazi].faces[iface] = AngFluxBCFace<RealType>(bc_sizes[iface], _npol, _ng);
            }
            nazi++;
        }
    }
    _old_angflux = _angflux;

    // Build angle weights
    _angle_weights.reserve(nazi);
    for (int iazi = 0; iazi < nazi; iazi++) {
        _angle_weights.push_back(std::vector<RealType>(_npol, static_cast<RealType>(0.0)));
        for (int ipol = 0; ipol < _angle_weights[iazi].size(); ipol++) {
            _angle_weights[iazi][ipol] = static_cast<RealType>(_ray_spacing[iazi] * azi_weights[iazi] * polar_weights[ipol]
                * M_PI * std::sin(polar_angles[ipol]));
        }
    }

    // Store the inverse polar angle sine
    _rsinpolang.resize(_npol);
    for (int ipol = 0; ipol < _npol; ipol++) {
        _rsinpolang[ipol] = static_cast<RealType>(1.0 / std::sin(polar_angles[ipol]));
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
            _segflux[j][i].resize(_ng, static_cast<RealType>(0.0));
        }
    }

    // Initialize the exponential argument array
    _exparg.resize(_max_segments + 1);
    for (size_t i = 0; i < _max_segments + 1; i++) {
        _exparg[i].resize(_ng, static_cast<RealType>(0.0));
    }
}

template <typename RealType>
void SerialMOC<RealType>::_read_rays() {
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
    
    // Apply ray sorting if requested
    if (_ray_sort == "long") {
        // Sort rays by number of segments (most segments first)
        std::sort(_rays.begin(), _rays.end(), 
                  [](const LongRay& a, const LongRay& b) {
                      return a._fsrs.size() > b._fsrs.size();
                  });
    } else if (_ray_sort == "short") {
        // Sort rays by number of segments (fewest segments first)  
        std::sort(_rays.begin(), _rays.end(),
                  [](const LongRay& a, const LongRay& b) {
                      return a._fsrs.size() < b._fsrs.size();
                  });
    }
    // If _ray_sort == "none", do nothing (default behavior)
    
    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << nrays << " rays from file: " << _filename << std::endl;
}

// Get the total cross sections for each FSR from the library
template <typename RealType>
void SerialMOC<RealType>::_get_xstr(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _xstr.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        auto transport_xs = library.total(fsr_mat_id[i]);
        _xstr[i].resize(transport_xs.size());
        for (size_t g = 0; g < transport_xs.size(); g++) {
            _xstr[i][g] = static_cast<RealType>(transport_xs[g]);
        }
    }
}

// Get the nu-fission cross sections for each FSR from the library
template <typename RealType>
void SerialMOC<RealType>::_get_xsnf(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _xsnf.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        auto nufiss_xs = library.nufiss(fsr_mat_id[i]);
        _xsnf[i].resize(nufiss_xs.size());
        for (size_t g = 0; g < nufiss_xs.size(); g++) {
            _xsnf[i][g] = static_cast<RealType>(nufiss_xs[g]);
        }
    }
}

// Get the chi for each FSR from the library
template <typename RealType>
void SerialMOC<RealType>::_get_xsch(
    const int num_fsr,
    const std::vector<int>& fsr_mat_id,
    const c5g7_library& library
) {
    _xsch.resize(num_fsr);
    for (auto i = 0; i < fsr_mat_id.size(); i++) {
        auto chi = library.chi(fsr_mat_id[i]);
        _xsch[i].resize(chi.size());
        for (size_t g = 0; g < chi.size(); g++) {
            _xsch[i][g] = static_cast<RealType>(chi[g]);
        }
    }
}

// Get the scattering cross sections for each FSR from the library
template <typename RealType>
void SerialMOC<RealType>::_get_xssc(
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
                _xssc[i][g][g2] = static_cast<RealType>(library.scat(fsr_mat_id[i], g, g2));
            }
        }
    }
}

// Reflect the angle for reflecting boundary conditions
inline int reflect_angle(int angle) {
    return angle % 2 == 0 ? angle + 1 : angle - 1;
}

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
template <typename RealType>
std::vector<double> SerialMOC<RealType>::fission_source(const double keff) const {
    std::vector<double> fissrc(_nfsr, 0.0);
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < _ng; g++) {
            fissrc[i] += _xsnf[i][g] * _scalar_flux[i][g] / keff;
        }
    }
    return fissrc;
}

// Build the total source term for each FSR based on the fission source and scattering cross sections
template <typename RealType>
void SerialMOC<RealType>::update_source(const std::vector<double>& fissrc) {
    int _nfsr = _scalar_flux.size();
    int ng = _scalar_flux[0].size();
    for (size_t i = 0; i < _nfsr; i++) {
        for (int g = 0; g < ng; g++) {
            _source[i][g] = static_cast<RealType>(fissrc[i]) * _xsch[i][g];
            for (int g2 = 0; g2 < ng; g2++) {
                if (g != g2) {
                    _source[i][g] += _xssc[i][g][g2] * static_cast<RealType>(_scalar_flux[i][g2]);
                }
            }
            _source[i][g] += _xssc[i][g][g] * static_cast<RealType>(_scalar_flux[i][g]);
            _source[i][g] /= (_xstr[i][g] * static_cast<RealType>(4.0 * M_PI));
        }
    }
}

// Main function to run the serial MOC sweep
template <typename RealType>
void SerialMOC<RealType>::sweep() {
    Kokkos::Profiling::pushRegion("SerialMOC::Sweep");
    // Initialize old values and a few scratch values
    int iseg, ireg, refl_angle;
    double phid1, phid2;

    // Initialize the scalar flux to 0.0
    for (auto i = 0; i < _nfsr; i++) {
        for (auto g = 0; g < _ng; g++) {
            _scalar_flux[i][g] = 0.0;
        }
    }

    // Sweep all the long rayse
    for (const auto& ray : _rays) {
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
                    : _old_angflux[ray.angle()].faces[ray._bc_face[0]]._angflux[ray._bc_index[0]][ipol][ig];
                _segflux[1][ray._fsrs.size()][ig] =
                    ray._bc_index[1] == -1
                    ? 0.0
                    : _old_angflux[ray.angle()].faces[ray._bc_face[1]]._angflux[ray._bc_index[1]][ipol][ig];
            }

            // Forward sweep
            for (iseg = 0; iseg < ray._fsrs.size(); iseg++) {
                ireg = ray._fsrs[iseg] - 1;
                for (size_t ig = 0; ig < _ng; ig++) {
                    phid1 = _segflux[RAY_START][iseg][ig] - _source[ireg][ig];
                    phid1 *= _exparg[iseg][ig];
                    _segflux[RAY_START][iseg + 1][ig] = _segflux[RAY_START][iseg][ig] - phid1;
                    _scalar_flux[ireg][ig] += phid1 * _angle_weights[ray.angle()][ipol];
                }
            }
            // Backward sweep
            for (iseg = ray._fsrs.size(); iseg > 0; iseg--) {
                ireg = ray._fsrs[iseg - 1] - 1;
                for (size_t ig = 0; ig < _ng; ig++) {
                    phid2 = _segflux[RAY_END][iseg][ig] - _source[ireg][ig];
                    phid2 *= _exparg[iseg - 1][ig];
                    _segflux[RAY_END][iseg - 1][ig] = _segflux[RAY_END][iseg][ig] - phid2;
                    _scalar_flux[ireg][ig] += phid2 * _angle_weights[ray.angle()][ipol];
                }
            }

            // Store the final segments' angular flux into the BCs
            for (size_t ig = 0; ig < _ng; ig++) {
                refl_angle = reflect_angle(ray.angle());
                if (ray._bc_index[RAY_END] != -1) {
                    _angflux[refl_angle].faces[ray._bc_face[RAY_END]]._angflux[ray._bc_index[RAY_END]][ipol][ig] =
                        _segflux[RAY_START][ray._fsrs.size()][ig];
                }
                refl_angle = reflect_angle(ray.angle());
                if (ray._bc_index[RAY_START] != -1) {
                    _angflux[refl_angle].faces[ray._bc_face[RAY_START]]._angflux[ray._bc_index[RAY_START]][ipol][ig] =
                        _segflux[RAY_END][0][ig];
                }
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

// Explicit template instantiations
template class SerialMOC<float>;
template class SerialMOC<double>;
