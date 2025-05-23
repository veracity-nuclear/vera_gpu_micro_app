# Attempt to locate HighFiveConfig.cmake first (modern CMake-compatible install)
find_package(HighFive CONFIG QUIET)

if(NOT HighFive_FOUND)
    # Fallback to manual find if CONFIG mode fails
    find_path(HighFive_INCLUDE_DIR HighFive/H5Easy.hpp)
    find_library(HighFive_LIBRARY NAMES highfive)

    if(HighFive_INCLUDE_DIR AND HighFive_LIBRARY)
        set(HighFive_FOUND TRUE)
        add_library(HighFive::HighFive UNKNOWN IMPORTED)
        set_target_properties(HighFive::HighFive PROPERTIES
            IMPORTED_LOCATION "${HighFive_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HighFive_INCLUDE_DIR}"
        )
    endif()
endif()
