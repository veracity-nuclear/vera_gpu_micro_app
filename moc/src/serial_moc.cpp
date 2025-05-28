#include "serial_moc.hpp"
#include <iostream>
#include <string>
#include <iomanip>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include "long_ray.hpp"
#include "c5g7_library.hpp"

// Reads all long rays from MPACT-generated HDF5 file
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

    // Set up the rays
    std::vector<LongRay> rays;
    rays.reserve(nrays);
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

// Build the fission source term for each FSR based on the scalar flux and nu-fission cross sections
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
        if (library.is_fissile(fsr_mat_id[i])) {
            for (int g = 0; g < ng; g++) {
                fissrc[i] += library.nufiss(fsr_mat_id[i], g) * scalar_flux[i][g] / keff;
                // std::cout << i << " " << g << " " << fissrc[i] << " " << library.nufiss(fsr_mat_id[i], g) / keff << " " << scalar_flux[i][g] << std::endl;
            }
        }
    }
    return fissrc;
}

// Build the total source term for each FSR based on the fission source and scattering cross sections
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

// Reflect the angle for reflecting boundary conditions
int reflect_angle(int angle) {
    return angle % 2 == 0 ? angle + 1 : angle - 1;
}

// Main function to run the serial MOC sweep
double serial_moc_sweep(const std::vector<std::string>& args) {
    if (args.size() != 3) {
        std::cerr << "Usage: " << args[0] << " <filename> <XS file>" << std::endl;
        return 1;
    }

    // Store the file names from the arguments
    std::string filename = args[1];
    std::string libname = args[2];

    // Process the file here
    HighFive::File file(filename, HighFive::File::ReadOnly);

    // Read the rays
    auto rays = read_rays(file);

    // Read mapping data
    auto xsrToFsrMap = file.getDataSet("/MOC_Ray_Data/Domain_00001/XSRtoFSR_Map").read<std::vector<int>>();
    auto starting_xsr = file.getDataSet("/MOC_Ray_Data/Domain_00001/Starting XSR").read<int>();

    // Adjust xsrToFsrMap by subtracting starting_xsr from each element
    for (auto& xsr : xsrToFsrMap) {
        xsr -= starting_xsr;
    }

    // Read the FSR volumes and plane height
    auto fsr_vol = file.getDataSet("/MOC_Ray_Data/Domain_00001/FSR_Volume").read<std::vector<double>>();
    auto pz = file.getDataSet("/MOC_Ray_Data/Domain_00001/plane_height").read<double>();
    int nfsr = fsr_vol.size();

    // Initialize the library
    c5g7_library library(libname);
    int ng = library.get_num_groups();

    // Read the material IDs
    auto tmp_mat_id = file.getDataSet("/MOC_Ray_Data/Domain_00001/Solution_Data/xsr_mat_id").read<std::vector<double>>();
    std::vector<int> xsr_mat_id;
    xsr_mat_id.reserve(tmp_mat_id.size());
    for (const auto& id : tmp_mat_id) {
        xsr_mat_id.push_back(static_cast<int>(id) - 1);
    }
    int nxsr = xsr_mat_id.size();

    // Calculate the FSR material IDs
    std::vector<int> fsr_mat_id(nfsr);
    int ixsr = 0;
    int nReg;
    for (int i = 0; i < nfsr; i++) {
        if (i == xsrToFsrMap[ixsr]) {
            ixsr++;
        }
        fsr_mat_id[i] = xsr_mat_id[ixsr - 1];
    }

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

    // Allocate scalar flux and source array
    double phid1, phid2, phio1, phio2;
    std::vector<std::vector<double>> scalar_flux(nfsr);
    for (size_t i = 0; i < nfsr; ++i) {
        scalar_flux[i].resize(ng, 1.0);
    }
    auto old_scalar_flux = scalar_flux;
    auto source = scalar_flux;

    // Read ray spacings and angular flux BC dimensions
    std::vector<AngFluxBCAngle> angflux;
    std::vector<double> ray_spacing;
    auto domain = file.getGroup("/MOC_Ray_Data/Domain_00001");
    auto polar_angles = file.getDataSet("/MOC_Ray_Data/Polar_Radians").read<std::vector<double>>();
    auto polar_weights = file.getDataSet("/MOC_Ray_Data/Polar_Weights").read<std::vector<double>>();
    auto azi_angles = file.getDataSet("/MOC_Ray_Data/Azimuthal_Radians").read<std::vector<double>>();
    auto azi_weights = file.getDataSet("/MOC_Ray_Data/Azimuthal_Weights").read<std::vector<double>>();
    int npol = polar_angles.size();
    int nazi = azi_angles.size();
    ray_spacing.clear();
    for (const auto& objName : domain.listObjectNames()) {
        // Loop over each angle group
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            nazi++;
            // Read ray spacing
            ray_spacing.push_back(angleGroup.getDataSet("spacing").read<double>());
            // Read the BC sizes
            int iazi = std::stoi(objName.substr(8)) - 1;
            auto bc_sizes = angleGroup.getDataSet("BC_size").read<std::vector<int>>();
            angflux.push_back(AngFluxBCAngle(4));
            for (size_t iface = 0; iface < 4; iface++) {
                angflux[iazi]._faces[iface] = AngFluxBCFace(bc_sizes[iface], npol, ng);
            }
        }
    }

    // Build angle weights
    std::vector<std::vector<double>> angle_weights;
    angle_weights.reserve(nazi);
    for (int iazi = 0; iazi < nazi; iazi++) {
        angle_weights.push_back(std::vector<double>(npol, 0.0));
        for (int ipol = 0; ipol < angle_weights[iazi].size(); ipol++) {
            angle_weights[iazi][ipol] = ray_spacing[iazi] * azi_weights[iazi] * polar_weights[ipol]
                * M_PI * std::sin(polar_angles[ipol]);
        }
    }

    // Store the inverse polar angle sine
    std::vector<double> rsinpolang(npol);
    for (int ipol = 0; ipol < npol; ipol++) {
        rsinpolang[ipol] = 1.0 / std::sin(polar_angles[ipol]);
    }

    // Initialize old values and a few scratch values
    double keff = 1.0;
    double old_keff = 1.0;
    std::vector<double> fissrc = build_fissrc(library, fsr_mat_id, scalar_flux, keff);
    std::vector<double> old_fissrc = fissrc;
    int iseg1, iseg2, ireg1, ireg2, refl_angle;
    auto old_angflux = angflux;
    std::vector<std::vector<double>> exparg(max_segments + 1);
    for (size_t i = 0; i < max_segments + 1; i++) {
        exparg[i].resize(ng, 0.0);
    }

    // Initialize iteration controls and print the header
    std::cout << "Iteration         keff       knorm      fnorm" << std::endl;
    double relaxation = 1.0;
    int max_iters = 10000;
    double kconv = 1.0e-8;
    double fconv = 1.0e-8;
    int debug_angle = 0;

    // Source iteration loop
    for (int iteration = 0; iteration < max_iters; iteration++) {
        // Build source and zero the fluxes
        source = build_source(library, fsr_mat_id, old_scalar_flux, fissrc);

        // Initialize the scalar flux to 0.0
        for (auto i = 0; i < nfsr; i++) {
            for (auto g = 0; g < ng; g++) {
                scalar_flux[i][g] = 0.0;
                // std::cout << "source " << i << " " << g << " " << source[i][g] << " " << old_scalar_flux[i][g] << std::endl;
            }
        }

        // Sweep all the long rayse
        for (const auto& ray : rays) {
            // if (ray.angle() == 2) {
            //     throw std::runtime_error("Beginning of ray loop");
            // }

            // Sweep all the polar angles
            for (size_t ipol = 0; ipol < npol; ipol++) {

                // Store the exponential arguments for this ray
                for (size_t i = 0; i < ray._fsrs.size(); i++) {
                    for (size_t ig = 0; ig < ng; ig++) {
                        exparg[i][ig] = 1.0 - std::exp(-xstr[ray._fsrs[i] - 1][ig] * ray._segments[i] * rsinpolang[ipol]);
                    }
                }

                // Initialize the ray flux with the angular flux BCs
                for (size_t ig = 0; ig < ng; ig++) {
                    segflux[0][0][ig] =
                        ray._bc_index[0] == -1
                        ? 0.0
                        : old_angflux[ray.angle()]._faces[ray._bc_face[0]]._angflux[ray._bc_index[0]][ipol][ig];
                    segflux[1][ray._fsrs.size()][ig] =
                        ray._bc_index[1] == -1
                        ? 0.0
                        : old_angflux[ray.angle()]._faces[ray._bc_face[1]]._angflux[ray._bc_index[1]][ipol][ig];
                }

                // Sweep the segments bi-directionally
                iseg2 = ray._fsrs.size();
                for (iseg1 = 0; iseg1 < ray._fsrs.size(); iseg1++) {
                    ireg1 = ray._fsrs[iseg1] - 1;
                    ireg2 = ray._fsrs[iseg2 - 1] - 1;

                    // Sweep the groups on the 2 segments
                    for (size_t ig = 0; ig < ng; ig++) {
                        // Forward segment sweep
                        phid1 = segflux[RAY_START][iseg1][ig] - source[ireg1][ig];
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " " << segflux[RAY_START][iseg1][ig] << " " << source[ireg1][ig] << " " << phid1 << std::endl;
                        // }
                        // TODO: tabulate exp
                        phid1 *= exparg[iseg1][ig];
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                        //         << 1.0 - std::exp(-xstr[ireg1][ig] * ray._segments[iseg1] * rsinpolang[ipol]) << " : " << phid1 << std::endl;
                        // }
                        segflux[RAY_START][iseg1 + 1][ig] = segflux[RAY_START][iseg1][ig] - phid1;
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                        //         << segflux[RAY_START][iseg1 + 1][ig] << " " << segflux[RAY_START][iseg1][ig] << " " << phid1
                        //         << std::endl;
                        // }
                        scalar_flux[ireg1][ig] += phid1 * angle_weights[ray.angle()][ipol];
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg1 << " " << ig << " "
                        //         << scalar_flux[ireg1][ig] << " " << phid1 << " " << angle_weights[ray.angle()][ipol] << std::endl;
                        // }

                        // Backward segment sweep
                        phid2 = segflux[RAY_END][iseg2][ig] - source[ireg2][ig];
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " " << segflux[RAY_END][iseg2][ig] << " " << source[ireg2][ig] << " " << phid2 << std::endl;
                        // }
                        phid2 *= exparg[iseg2 - 1][ig];
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " "
                        //     << 1.0 - std::exp(-xstr[ireg2][ig] * ray._segments[iseg2 - 1] * rsinpolang[ipol]) << " "
                        //     << " : " << phid2 << std::endl;
                        // }
                        segflux[RAY_END][iseg2 - 1][ig] = segflux[RAY_END][iseg2][ig] - phid2;
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " "
                        //         << segflux[RAY_END][iseg2 - 1][ig] << " " << segflux[RAY_END][iseg2][ig] << " " << phid2 << std::endl;
                        // }
                        scalar_flux[ireg2][ig] += phid2 * angle_weights[ray.angle()][ipol];
                        // if (ray.angle() == debug_angle) {
                        //     std::cout << ray.angle() << " " << ipol << " " << iseg2 << " " << ig << " "
                        //         << scalar_flux[ireg2][ig] << " " << phid2 << " " << angle_weights[ray.angle()][ipol] << std::endl;
                        // }
                    }
                    // throw std::runtime_error("end of first segment");
                    iseg2--;
                }

                // Store the final segments' angular flux into the BCs
                for (size_t ig = 0; ig < ng; ig++) {
                    refl_angle = reflect_angle(ray.angle());
                    if (ray._bc_index[RAY_START] != -1) {
                        angflux[refl_angle]._faces[ray._bc_face[RAY_END]]._angflux[ray._bc_index[RAY_END]][ipol][ig] =
                            segflux[RAY_START][iseg1][ig];
                    }
                    refl_angle = reflect_angle(ray.angle());
                    if (ray._bc_index[RAY_END] != -1) {
                        angflux[refl_angle]._faces[ray._bc_face[RAY_START]]._angflux[ray._bc_index[RAY_START]][ipol][ig] =
                            segflux[RAY_END][0][ig];
                    }
                }
                // throw std::runtime_error("End of segment loop");
            }
            // throw std::runtime_error("End of polar loop");
        }

        // Scale the flux with source, volume, and transport XS
        for (size_t i = 0; i < nfsr; ++i) {
            for (size_t g = 0; g < ng; ++g) {
                // std::cout << i << " " << g << " scale " << scalar_flux[i][g] << " " << xstr[i][g] << " " << fsr_vol[i] << " " << pz << " " << source[i][g] << " " << 4.0 * M_PI << std::endl;
                scalar_flux[i][g] = scalar_flux[i][g] / (xstr[i][g] * fsr_vol[i] / pz) + source[i][g] * 4.0 * M_PI;
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
            if (library.is_fissile(fsr_mat_id[i])) {
                for (size_t g = 0; g < ng; ++g) {
                    numerator += scalar_flux[i][g] * library.nufiss(fsr_mat_id[i], g) * fsr_vol[i];
                    denominator += old_scalar_flux[i][g] * library.nufiss(fsr_mat_id[i], g) * fsr_vol[i];
                }
            }
        }
        keff = old_keff * numerator / denominator;

        // Calculate fission source convergence metric
        double fnorm = 0.0;
        for (size_t i = 0; i < scalar_flux.size(); ++i) {
            for (size_t g = 0; g < scalar_flux[i].size(); ++g) {
                fnorm += (scalar_flux[i][g] - old_scalar_flux[i][g]) * library.nufiss(fsr_mat_id[i], g) * fsr_vol[i];
            }
        }
        fnorm = sqrt(fnorm * fnorm / double(fissrc.size()));

        // Calculate the keff convergence metric
        double knorm = keff - old_keff;

        // Print the iteration results
        std::cout << " " << std::setw(8) << iteration
                  << "   " << std::fixed << std::setprecision(8) << keff
                  << "   " << std::scientific << std::setprecision(2) << knorm
                  << "   " << fnorm << std::endl;

        // Check for convergence
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
        old_angflux = angflux;
    }

    return keff;
}