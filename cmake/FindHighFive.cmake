# Fallback CMake module to locate HighFive and export a proper interface target
# Hardcode include path to /opt/highfive/include

# Locate HDF5 (required dependency)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# Create an INTERFACE imported target for HighFive
add_library(HighFive::HighFive INTERFACE IMPORTED)
set_target_properties(HighFive::HighFive PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "/opt/highfive/include"
)

# Link HDF5 imported targets to HighFive interface
# This ensures HDF5 symbols (e.g., H5Treclaim) are linked transitively
target_link_libraries(HighFive::HighFive INTERFACE
    HDF5::HDF5_C
    HDF5::HDF5_HL
)
