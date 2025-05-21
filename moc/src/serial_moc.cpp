#include <iostream>
#include <string>
#include <H5Cpp.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::cout << "File name: " << filename << std::endl;

    // Process the file here
    try {
        // Open the HDF5 file
        H5::H5File file(filename, H5F_ACC_RDONLY);
        std::cout << "Successfully opened HDF5 file: " << filename << std::endl;

        // File processing code would go here

        // Close the file when done
        file.close();
    } catch (H5::Exception& e) {
        std::cerr << "Error opening HDF5 file: " << e.getCDetailMsg() << std::endl;
        return 1;
    }

    return 0;
}