#include <iostream>
#include <string>
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
    const std::vector<int>& xsrToFsrMap,
    const std::vector<int>& xsr_mat_id,
    const c5g7_library& library
) {
    std::vector<std::vector<double>> xs;
    xs.resize(num_fsr);
    for (auto i = 0; i < xsrToFsrMap.size(); i++) {
        auto starting_fsr = xsrToFsrMap[i] - 1;
        auto stopping_fsr = i == xsrToFsrMap.size() - 1 ? num_fsr - 1 : xsrToFsrMap[i + 1] - 1;
        for (auto j = starting_fsr; j < stopping_fsr + 1; j++) {
            xs[j] = library.abs(xsr_mat_id[i]);
        }
    }
    return xs;
}

std::vector<std::vector<double>> build_source(
    const c5g7_library& library,
    const std::vector<std::vector<double>>& scalar_flux
) {
    int nfsr = scalar_flux.size();
    int ng = scalar_flux[0].size();
    std::vector<std::vector<double>> source;
    source.resize(nfsr);
    for (size_t i = 0; i < nfsr; i++) {
        source[i].resize(ng);
        double fissrc = 0.0;
        for (int g = 0; g < ng; g++) {
            fissrc += library.nufiss(i, g) * scalar_flux[i][g];
        }
        for (int g = 0; g < ng; g++) {
            source[i][g] = fissrc * library.chi(i, g);
            for (int g2 = 0; g2 < ng; g2++) {
                source[i][g] += library.scat(i, g, g2) * scalar_flux[i][g2];
            }
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
    auto xsr_mat_id = std::vector<int>(tmp_mat_id.begin(), tmp_mat_id.end());
    int nxsr = xsr_mat_id.size();

    // Initialize the library
    c5g7_library library("../data/c5g7.xsl");

    // Get XS
    auto xstr = get_xstr(nfsr, starting_xsr, xsrToFsrMap, xsr_mat_id, library);

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
        std::fill(scalar_flux[i].begin(), scalar_flux[i].end(), 1.0);
    }
    auto old_scalar_flux = scalar_flux;

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

    for (int iteration = 0; iteration < 10; iteration++) {

        // Build source
        auto source = build_source(library, old_scalar_flux);

        // Sweep
        for (const auto& ray : rays) {
            // Initialize the angular flux to 1.0
            for (size_t ig = 0; ig < ng; ig++) {
                segflux[0][0][ig] = 0.0;
                segflux[1][max_segments][ig] = 0.0;
            }
            // Sweep the segments
            int iseg2 = ray._fsrs.size();
            for (int iseg1 = 0; iseg1 < ray._fsrs.size(); iseg1++) {
                iseg2--;
                int ireg1 = ray._fsrs[iseg1] - 1;
                int ireg2 = ray._fsrs[iseg2] - 1;
                // Sweep the groups
                for (size_t ipol = 0; ipol < 1; ipol++) {
                    for (size_t ig = 0; ig < ng; ig++) {
                        phid1 = segflux[0][iseg1][ig] - source[ireg1][ig];
                        // TODO: tabulate exp
                        phid1 *= std::exp(-xstr[ireg1][ig] * ray._segments[iseg1] * angle_weights[ray.angle()][ipol]);
                        // TODO: use real weight
                        segflux[0][iseg1 + 1][ig] = segflux[0][iseg1][ig] + phid1 * 0.5;
                        scalar_flux[ireg1][ig] += phid1 * 0.5;

                        phid2 = segflux[1][iseg2 + 1][ig] - source[ireg2][ig];
                        phid2 *= std::exp(-xstr[ireg2][ig] * ray._segments[iseg2 + 1] * angle_weights[ray.angle()][ipol]);
                        segflux[1][iseg2][ig] = segflux[0][iseg2 + 1][ig] + phid2 * 0.5;
                        scalar_flux[ireg2][ig] += phid2 * 0.5;
                    }
                }
            }
        }

        // Print the scalar flux
        std::cout << "Scalar Flux:" << std::endl;
        for (size_t i = 0; i < scalar_flux.size(); ++i) {
            std::cout << "FSR " << i << ": ";
            for (size_t g = 0; g < scalar_flux[i].size(); ++g) {
                std::cout << scalar_flux[i][g] << " ";
            }
            std::cout << std::endl;
        }

        // Save the old scalar flux
        for (size_t i = 0; i < nfsr; ++i) {
            for (size_t g = 0; g < ng; ++g) {
                old_scalar_flux[i][g] = 0.5 * old_scalar_flux[i][g] + 0.5 * scalar_flux[i][g];
            }
        }
    }

    return 0;
}