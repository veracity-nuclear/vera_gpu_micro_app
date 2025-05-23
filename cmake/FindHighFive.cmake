# Fallback CMake module to locate HighFive and export a proper target

# Locate HighFive headers
find_path(HighFive_INCLUDE_DIR
  NAMES highfive/H5File.hpp
  PATHS
    $ENV{HIGHFIVE_ROOT}
    ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES include
  DOC "Path to HighFive include dir"
)

# Locate HDF5 (required dependency)
find_package(HDF5 REQUIRED COMPONENTS C HL)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HighFive REQUIRED_VARS HighFive_INCLUDE_DIR)

if(HighFive_FOUND AND NOT TARGET HighFive::HighFive)
  add_library(HighFive::HighFive INTERFACE IMPORTED)
  set_target_properties(HighFive::HighFive PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${HighFive_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES};${HDF5_HL_LIBRARIES}"
  )
endif()