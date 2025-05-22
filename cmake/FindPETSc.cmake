# Attempt to find PETScConfig.cmake first (modern CMake-compatible install)
find_package(PETSc CONFIG QUIET)

if(NOT PETSc_FOUND)
    # Get PETSC_DIR and PETSC_ARCH from environment if not passed explicitly
    if(NOT DEFINED PETSC_DIR AND DEFINED ENV{PETSC_DIR})
        set(PETSC_DIR $ENV{PETSC_DIR})
    endif()

    if(NOT DEFINED PETSC_ARCH AND DEFINED ENV{PETSC_ARCH})
        set(PETSC_ARCH $ENV{PETSC_ARCH})
    endif()

    if(NOT DEFINED PETSC_DIR)
        message(FATAL_ERROR "PETSC_DIR must be defined (via -DPETSC_DIR=... or environment)")
    endif()

    # Determine include/lib path layout
    if(PETSC_ARCH AND EXISTS "${PETSC_DIR}/${PETSC_ARCH}/include")
        set(PETSC_INCLUDE_DIR "${PETSC_DIR}/${PETSC_ARCH}/include")
        set(PETSC_LIBRARY_DIR "${PETSC_DIR}/${PETSC_ARCH}/lib")
    elseif(EXISTS "${PETSC_DIR}/include" AND EXISTS "${PETSC_DIR}/lib")
        set(PETSC_INCLUDE_DIR "${PETSC_DIR}/include")
        set(PETSC_LIBRARY_DIR "${PETSC_DIR}/lib")
    else()
        message(FATAL_ERROR "PETSc could not be found in either flat or arch-specific layout under: ${PETSC_DIR}")
    endif()

    # Locate PETSc library and header
    find_path(PETSC_INCLUDE_DIR_FINAL petsc.h PATHS "${PETSC_INCLUDE_DIR}")
    find_library(PETSC_LIBRARY petsc PATHS "${PETSC_LIBRARY_DIR}")

    if(PETSC_INCLUDE_DIR_FINAL AND PETSC_LIBRARY)
        set(PETSc_FOUND TRUE)

        add_library(PETSc::PETSc UNKNOWN IMPORTED)
        set_target_properties(PETSc::PETSc PROPERTIES
            IMPORTED_LOCATION "${PETSC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSC_INCLUDE_DIR_FINAL}"
        )
    else()
        message(FATAL_ERROR "Could not locate PETSc headers or library under ${PETSC_DIR}")
    endif()
endif()
