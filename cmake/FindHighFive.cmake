# Fallback CMake module to locate HighFive and export a proper interface target
# Hardcode include path to /opt/highfive/include

# Locate HDF5 (required dependency)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# Create an INTERFACE imported target for HighFive
add_library(HighFive::HighFive INTERFACE IMPORTED)
set_target_properties(HighFive::HighFive PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "/opt/highfive/include"
    INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES}"
)
