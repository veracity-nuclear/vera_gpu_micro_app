# Fallback CMake module to locate HighFive and export an interface target
# Hardcode include path to /opt/highfive/include

# Locate HDF5 (required dependency)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# Create an INTERFACE imported target for HighFive
add_library(HighFive::HighFive INTERFACE IMPORTED)

# Specify include directories
target_include_directories(HighFive::HighFive INTERFACE
  "/opt/highfive/include"
)

# Link both HDF5 C and high-level libraries using traditional variables
target_link_libraries(HighFive::HighFive INTERFACE
  ${HDF5_C_LIBRARIES}
  ${HDF5_HL_LIBRARIES}
)