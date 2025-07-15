#include "serial_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include "long_ray.hpp"
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
                _xstr[i][to] = xstr[ixsr - 1][to];
                _xsnf[i][to] = xstr[ixsr - 1][to];
                _xsch[i][to] = xstr[ixsr - 1][to];
                _xssc[i][to].resize(_ng);
                for (int from = 0; from < _ng; from++) {
                    _xssc[i][to][from] = xssc[ixsr - 1][to][from];
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
                _angflux[iazi].faces[iface] = AngFluxBCFace(bc_sizes[iface], _npol, _ng);
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

    // Initialize inline exponential tables
    // Set up tables similar to MPACT: polar angle dependent linear interpolation
    // Table covers range [-40, 0] with 40000 intervals
    int n_intervals = 40000;
    double min_val = -40.0;
    double max_val = 0.0;
    double dx = (max_val - min_val) / n_intervals;

    _expoa.resize(n_intervals + 1);
    _expob.resize(n_intervals + 1);

    for (int i = 0; i <= n_intervals; i++) {
        _expoa[i].resize(_npol);
        _expob[i].resize(_npol);

        double x1 = min_val + i * dx;
        double x2 = x1 + dx;
        double y1 = 1.0 - std::exp(x1);
        double y2 = 1.0 - std::exp(x2);

        for (int ipol = 0; ipol < _npol; ipol++) {
            // Scale by polar angle sine factor like MPACT does
            double rpol = _rsinpolang[ipol];
            double x1_scaled = x1 * rpol;
            double x2_scaled = x2 * rpol;
            double y1_scaled = 1.0 - std::exp(x1_scaled);
            double y2_scaled = 1.0 - std::exp(x2_scaled);

            // Linear interpolation coefficients: y = m*x + b
            _expoa[i][ipol] = (y2_scaled - y1_scaled) / dx;  // slope
            _expob[i][ipol] = y1_scaled - _expoa[i][ipol] * x1;  // intercept
            _expoa[i][ipol] *= 0.001;
        }
    }

    // Scale segment lengths by 1000.0 to avoid repeated multiplication in sweep
    for (auto& ray : _rays) {
        for (auto& segment : ray._segments) {
            segment *= 1000.0;
        }
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
                    double xval = -_xstr[ray._fsrs[i] - 1][ig] * ray._segments[i];
                    int ix = static_cast<int>(std::floor(xval)) + 40000;
                    ix = std::max(ix, -40000);  // Clamp to table bounds
                    ix = std::min(ix, 40000);
                    _exparg[i][ig] = _expoa[ix][ipol] * xval + _expob[ix][ipol];
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
