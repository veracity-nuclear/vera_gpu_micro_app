# Fallback CMake module to locate HighFive and export an interface target

# Locate HDF5 (required dependency)
find_package(HDF5 REQUIRED COMPONENTS C HL)

# Try to find the main HighFive header
find_path(HIGHFIVE_INCLUDE_DIR
  NAMES highfive/H5File.hpp
  HINTS
    ENV HIGHFIVE_ROOT
    ${CMAKE_PREFIX_PATH}
    /opt/highfive
  PATH_SUFFIXES include
)

# Create an INTERFACE imported target for HighFive
add_library(HighFive::HighFive INTERFACE IMPORTED)

# Specify include directories
target_include_directories(HighFive::HighFive INTERFACE
  ${HIGHFIVE_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
)

# Link both HDF5 C and high-level libraries using traditional variables
target_link_libraries(HighFive::HighFive INTERFACE
  ${HDF5_C_LIBRARIES}
  ${HDF5_HL_LIBRARIES}
)