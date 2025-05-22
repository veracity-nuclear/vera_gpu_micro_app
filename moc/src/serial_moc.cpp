#include <iostream>
#include <string>
#include <iomanip>
#include <H5Cpp.h>
#include "highfive/highfive.hpp"
#include "long_ray.hpp"
#include "c5g7_library.hpp"
#include "quadrature.hpp"

std::vector<LongRay> read_rays(HighFive::File file) {
    auto domain = file.getGroup("/MOC_Ray_Data/Domain_00001");

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
    nangles /= 2;

    // Set up the rays
    std::vector<LongRay> rays;
    rays.reserve(nrays);
    nrays = 0;
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);

            // Read the radians data from the angle group
            auto radians = angleGroup.getDataSet("Radians").read<double>();
            auto angleIndex = (std::stoi(objName.substr(8)) - 1) % nangles;
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    auto rayGroup = angleGroup.getGroup(rayName);
                    rays.push_back(LongRay(rayGroup, angleIndex, radians));
                    nrays++;
                }
            }
        }
    }
    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << nrays << " rays from file: " << file.getName() << std::endl;
    return rays;
}

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

std::vector<double> build_fissrc(
    const c5g7_library& library,
    const std::vector<int>& fsr_mat_id,
    const std::vector<std::vector<double>>& scalar_flux,
    double keff
) {
    int nfsr = scalar_flux.size();
    int ng = scalar_flux[0].size();
    std::vector<double> fissrc(nfsr, 0.0);
    int ixsr = 1;
    for (size_t i = 0; i < nfsr; i++) {
        for (int g = 0; g < ng; g++) {
            fissrc[i] += library.nufiss(fsr_mat_id[i], g) * scalar_flux[i][g] / keff;
        }
    }
    return fissrc;
}

std::vector<std::vector<double>> build_source(
    const c5g7_library& library,
    const std::vector<int>& fsr_mat_id,
    const std::vector<std::vector<double>>& scalar_flux,
    const std::vector<double>& fissrc
) {
    int nfsr = scalar_flux.size();
    int ng = scalar_flux[0].size();
    std::vector<std::vector<double>> source;
    source.resize(nfsr);
    int ixsr = 1;
    for (size_t i = 0; i < nfsr; i++) {
        source[i].resize(ng);
        for (int g = 0; g < ng; g++) {
            source[i][g] = fissrc[i] * library.chi(fsr_mat_id[i], g);
            // std::cout << "mgfs " << i << " " << g << " " << i << " " << fissrc[i] << " " << library.chi(fsr_mat_id[i], g) << " : " << source[i][g] << std::endl;
            for (int g2 = 0; g2 < ng; g2++) {
                if (g != g2) {
                    source[i][g] += library.scat(fsr_mat_id[i], g, g2) * scalar_flux[i][g2];
                    // std::cout << "inscatter " << i << " " << g << " " << g2 << " " << " " << scalar_flux[i][g2] << " " << library.scat(fsr_mat_id[i], g, g2) << " : " << source[i][g] << std::endl;
                }
            }
            double old_source = source[i][g];
            source[i][g] += library.self_scat(fsr_mat_id[i], g) * scalar_flux[i][g];
            // std::cout << "selfscatter a " << g << " " << i << " " << old_source << " " << library.self_scat(fsr_mat_id[i], g) << " " << scalar_flux[i][g] << " : " << source[i][g] << std::endl;
            source[i][g] /= (library.total(fsr_mat_id[i], g) * 4.0 * M_PI);
            // std::cout << "selfscatter b " << g << " " << i << " " << 4.0 * M_PI << " " << library.total(fsr_mat_id[i], g) <<  " : " << source[i][g] << std::endl;
        }
    }
    return source;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // Process the file here
    HighFive::File file(filename, HighFive::File::ReadOnly);

    // Read the rays
    auto rays = read_rays(file);

    // Read mapping data
    auto xsrToFsrMap = file.getDataSet("/MOC_Ray_Data/Domain_00001/XSRtoFSR_Map").read<std::vector<int>>();
    auto starting_xsr = file.getDataSet("/MOC_Ray_Data/Domain_00001/Starting XSR").read<int>();
    auto xsr_vol = file.getDataSet("/MOC_Ray_Data/Domain_00001/XSR_Volume").read<std::vector<double>>();
    auto pz = file.getDataSet("/MOC_Ray_Data/Domain_00001/plane_height").read<double>();

    // Read solution data
    auto temp_fsr_flux = file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/fsr_flux").read<std::vector<std::vector<double>>>();
    std::vector<std::vector<double>> fsr_flux;
    if (!temp_fsr_flux.empty()) {
        size_t num_groups = temp_fsr_flux[0].size();
        size_t num_fsrs = temp_fsr_flux.size();
        fsr_flux.resize(num_groups, std::vector<double>(num_fsrs));
        for (size_t i = 0; i < num_fsrs; ++i) {
            for (size_t j = 0; j < num_groups; ++j) {
                fsr_flux[j][i] = temp_fsr_flux[i][j];
            }
        }
    }
    int nfsr = fsr_flux.size();
    int ng = fsr_flux[0].size();

    auto tmp_mat_id = file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xsr_mat_id").read<std::vector<double>>();
    std::vector<int> xsr_mat_id;
    xsr_mat_id.reserve(tmp_mat_id.size());
    for (const auto& id : tmp_mat_id) {
        xsr_mat_id.push_back(static_cast<int>(id) - 1);
    }
    int nxsr = xsr_mat_id.size();

    std::vector<int> fsr_mat_id(nfsr);
    std::vector<double> vol(nfsr);
    int ixsr = 1;
    for (int i = 0; i < nfsr; i++) {
        int index;
        if (ixsr == xsrToFsrMap.size()) {
            index = ixsr - 1;
        } else if (i == xsrToFsrMap[ixsr]) {
            index = ixsr;
            ixsr++;
        } else {
            index = ixsr - 1;
        }
        fsr_mat_id[i] = xsr_mat_id[index];
        vol[i] = xsr_vol[index];
    }

    // Initialize the library
    c5g7_library library("../data/c5g7.xsl");

    // Get XS
    auto xstr = get_xstr(nfsr, starting_xsr, fsr_mat_id, library);

    // Allocate segment flux array
    std::vector<std::vector<std::vector<double>>> segflux;
    size_t max_segments = 0;
    for (const auto& ray : rays) {
        max_segments = std::max(max_segments, ray._fsrs.size());
    }
    segflux.resize(2);
    for (size_t j = 0; j < 2; j++) {
        segflux[j].resize(max_segments + 1);
        for (size_t i = 0; i < max_segments + 1; i++) {
            segflux[j][i].resize(ng, 0.0);
        }
    }

    // Allocate scalar flux array
    double phid1, phid2, phio1, phio2;
    std::vector<std::vector<double>> scalar_flux = fsr_flux;
    for (size_t i = 0; i < nfsr; ++i) {
        std::fill(scalar_flux[i].begin(), scalar_flux[i].end(), 1.17);
    }
    auto old_scalar_flux = scalar_flux;
    auto source = scalar_flux;

    // Quadrature
    Quadrature quadrature = Quadrature(1, 1);

    // Read ray spacings
    std::vector<double> ray_spacing;
    auto domain = file.getGroup("/MOC_Ray_Data/Domain_00001");
    ray_spacing.clear();
    for (const auto& objName : domain.listObjectNames()) {
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            ray_spacing.push_back(angleGroup.getDataSet("spacing").read<double>());
        }
    }

    // If no spacing found, use a default value
    if (ray_spacing.empty()) {
        std::cout << "Warning: No ray spacing data found, using default value of 0.03" << std::endl;
        ray_spacing.push_back(0.03);
    }

    // Build weights
    std::vector<std::vector<double>> angle_weights;
    angle_weights.resize(1);
    for (int iazi = 0; iazi < angle_weights.size(); iazi++) {
        angle_weights[iazi].resize(1);
        for (int ipol = 0; ipol < angle_weights[iazi].size(); ipol++) {
            angle_weights[iazi][ipol] = ray_spacing[iazi] * quadrature.azi_weight(iazi) * quadrature.pol_weight(ipol)
                * M_PI * std::sin(quadrature.pol_angle(ipol));
        }
    }
    std::vector<double> rsinpolang(quadrature.npol());
    for (int ipol = 0; ipol < quadrature.npol(); ipol++) {
        rsinpolang[ipol] = 1.0 / std::sin(quadrature.pol_angle(ipol));
    }

    // Miscellaneous
    double keff = 1.0;
    double old_keff = 1.0;
    std::vector<double> fissrc = build_fissrc(library, fsr_mat_id, scalar_flux, keff);
    std::vector<double> old_fissrc = fissrc;
    int iseg1, iseg2, ireg1, ireg2;

    std::cout << "Iteration         keff       knorm      fnorm" << std::endl;
    double relaxation = 1.0;
    int max_iters = 1000;
    double kconv = 1.0e-8;
    double fconv = 1.0e-8;
    for (int iteration = 0; iteration < max_iters; iteration++) {

        // Build source and zero the fluxes
        source = build_source(library, fsr_mat_id, old_scalar_flux, fissrc);
        for (auto i = 0; i < nfsr; i++) {
            for (auto g = 0; g < ng; g++) {
                scalar_flux[i][g] = 0.0;
                // std::cout << "source " << i << " " << g << " " << source[i][g] << " " << old_scalar_flux[i][g] << std::endl;
            }
        }

        // Sweep
        for (const auto& ray : rays) {
            for (size_t ipol = 0; ipol < 1; ipol++) {
                // Initialize the angular flux to 1.0
                for (size_t ig = 0; ig < ng; ig++) {
                    segflux[0][0][ig] = 0.0;
                    segflux[1][ray._fsrs.size()][ig] = 0.0;
                }
                // Sweep the segments
                iseg2 = ray._fsrs.size();
                for (iseg1 = 0; iseg1 < ray._fsrs.size(); iseg1++) {
                    ireg1 = ray._fsrs[iseg1] - 1;
                    ireg2 = ray._fsrs[iseg2 - 1] - 1;
                    // Sweep the groups
                    for (size_t ig = 0; ig < ng; ig++) {
                        phid1 = segflux[0][iseg1][ig] - source[ireg1][ig];
                        // std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " " << segflux[0][iseg1][ig] << " " << source[ireg1][ig] << " " << phid1 << std::endl;
                        // TODO: tabulate exp
                        phid1 *= 1.0 - std::exp(-xstr[ireg1][ig] * ray._segments[iseg1] * rsinpolang[ipol]);
                        // std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                        // << " : " << phid1 << std::endl;
                        segflux[0][iseg1 + 1][ig] = segflux[0][iseg1][ig] - phid1;
                        // std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                        //     << segflux[0][iseg1 + 1][ig] << " " << segflux[0][iseg1][ig] << " " << phid1
                        //     << std::endl;
                        scalar_flux[ireg1][ig] += phid1 * angle_weights[ray.angle()][ipol];
                        // std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                        //     << scalar_flux[ireg1][ig] << " " << phid1 << " " << angle_weights[ray.angle()][ipol] << std::endl;

                        phid2 = segflux[1][iseg2][ig] - source[ireg2][ig];
                        phid2 *= 1.0 - std::exp(-xstr[ireg2][ig] * ray._segments[iseg2 - 1] * rsinpolang[ipol]);
                        segflux[1][iseg2 - 1][ig] = segflux[0][iseg2][ig] - phid2;
                        scalar_flux[ireg2][ig] += phid2 * angle_weights[ray.angle()][ipol];
                        // std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " "
                        //     << scalar_flux[ireg2][ig] << " " << phid2 << " " << angle_weights[ray.angle()][ipol] << std::endl;
                    }
                    // throw std::runtime_error("Not implemented: segflux[0][iseg1 + 1][ig] = segflux[0][iseg1][ig] + phid1 * 0.5;");
                    iseg2--;
                }
            }
        }

        // Scale the flux
        for (size_t i = 0; i < nfsr; ++i) {
            for (size_t g = 0; g < ng; ++g) {
                // std::cout << "scale " << scalar_flux[i][g] << " " << xstr[i][g] << " " << vol[i] << " " << pz << " " << source[i][g] << " " << 4.0 * M_PI << std::endl;
                scalar_flux[i][g] = scalar_flux[i][g] / (xstr[i][g] * vol[i] / pz) + source[i][g] * 4.0 * M_PI;
            }
        }

        // Print the scalar flux
        // std::cout << "Scalar Flux:" << std::endl;
        // for (size_t i = 0; i < scalar_flux.size(); ++i) {
        //     std::cout << "FSR " << i << ": ";
        //     for (size_t g = 0; g < scalar_flux[i].size(); ++g) {
        //         std::cout << scalar_flux[i][g] << " ";
        //     }
        //     std::cout << std::endl;
        // }

        // Update fission source and keff
        fissrc = build_fissrc(library, fsr_mat_id, scalar_flux, keff);
        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t i = 0; i < nfsr; ++i) {
            numerator += fissrc[i] * vol[i];
            denominator += old_fissrc[i] * vol[i];
        }
        keff = old_keff * numerator / denominator / double(vol.size());

        // Calculate fission source convergence metric
        double fnorm = 0.0;
        for (size_t i = 0; i < scalar_flux.size(); ++i) {
            for (size_t g = 0; g < scalar_flux[i].size(); ++g) {
                fnorm += (scalar_flux[i][g] - old_scalar_flux[i][g]) * library.nufiss(fsr_mat_id[i], g) * vol[i];
            }
        }
        double knorm = keff - old_keff;
        fnorm = sqrt(fnorm * fnorm / double(fissrc.size()));
        std::cout << " " << std::setw(8) << iteration
                  << "   " << std::fixed << std::setprecision(8) << keff
                  << "   " << std::scientific << std::setprecision(2) << knorm
                  << "   " << fnorm << std::endl;
        if (fabs(knorm) < kconv && fabs(fnorm) < fconv) {
            std::cout << "Converged after " << iteration + 1 << " iterations." << std::endl;
            break;
        }

        // Save the old values
        for (size_t i = 0; i < nfsr; ++i) {
            for (size_t g = 0; g < ng; ++g) {
                old_scalar_flux[i][g] = relaxation * scalar_flux[i][g] + (1.0 - relaxation) * old_scalar_flux[i][g];
            }
        }
        old_keff = keff;
        fissrc = build_fissrc(library, fsr_mat_id, scalar_flux, keff);
        old_fissrc = fissrc;
    }

    return 0;
}