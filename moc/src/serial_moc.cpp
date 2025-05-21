#include <iostream>
#include <string>
#include <H5Cpp.h>
#include "highfive/highfive.hpp"
#include "long_ray.hpp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // Process the file here
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::Group domain = file.getGroup("/MOC_Ray_Data/Domain_00001");

    // Count the rays
    int nrays = 0;
    for (size_t i = 0; i < domain.listObjectNames().size(); i++) {
        std::string objName = domain.listObjectNames()[i];
        if (objName.substr(0, 6) == "Angle_") {
            HighFive::Group angleGroup = domain.getGroup(objName);
            std::vector<std::string> angleObjects = angleGroup.listObjectNames();
            for (const auto& rayName : angleObjects) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    nrays++;
                }
            }
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
            double radians = angleGroup.getDataSet("Radians").read<double>();
            for (const auto& rayName : angleGroup.listObjectNames()) {
                if (rayName.substr(0, 8) == "LongRay_") {
                    HighFive::Group rayGroup = angleGroup.getGroup(rayName);
                    rays.push_back(LongRay(rayGroup, radians));
                    nrays++;
                }
            }
        }
    }
    // Print a message with the number of rays and filename
    std::cout << "Successfully set up " << nrays << " rays from file: " << filename << std::endl;

    // Read mapping data
    std::vector<int> xsrToFsrMap = domain.getDataSet("XSRtoFSR_Map").read<std::vector<int>>();
    int starting_xsr = domain.getDataSet("Starting XSR").read<int>();

    // Read solution data
    std::vector<std::vector<double>> fsr_flux = domain.getDataSet("Solution_Data/fsr_flux").read<std::vector<std::vector<double>>>();
    std::vector<long> xsr_mat_id = domain.getDataSet("Solution_Data/xsr_mat_id").read<std::vector<long>>();

    return 0;
}