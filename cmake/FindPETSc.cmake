# Try config-mode first (preferred if PETScConfig.cmake exists)
find_package(PETSc CONFIG QUIET)

if(NOT PETSc_FOUND)
    # Try to get PETSC_DIR and PETSC_ARCH from environment or cache
    if(DEFINED ENV{PETSC_DIR})
        set(PETSC_DIR "$ENV{PETSC_DIR}")
    endif()

    if(DEFINED ENV{PETSC_ARCH})
        set(PETSC_ARCH "$ENV{PETSC_ARCH}")
    endif()

    if(NOT DEFINED PETSC_DIR)
        message(FATAL_ERROR "PETSC_DIR must be defined via environment or -DPETSC_DIR=...")
    endif()

    # Use PETSC_ARCH if provided, otherwise fallback to flat layout
    if(DEFINED PETSC_ARCH)
        set(_petsc_root "${PETSC_DIR}/${PETSC_ARCH}")
    else()
        set(_petsc_root "${PETSC_DIR}")
    endif()

    # Include both base and arch-specific include directories
    find_path(PETSC_INCLUDE_DIR petsc.h
        PATHS
            ${_petsc_root}/include
            ${PETSC_DIR}/include
            ${PETSC_DIR}/${PETSC_ARCH}/include
    )

    find_library(PETSC_LIBRARY petsc
        PATHS ${_petsc_root}/lib
    )

    if(PETSC_INCLUDE_DIR AND PETSC_LIBRARY)
        set(PETSc_FOUND TRUE)
        add_library(PETSc::PETSc UNKNOWN IMPORTED)
        set_target_properties(PETSc::PETSc PROPERTIES
            IMPORTED_LOCATION "${PETSC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PETSC_INCLUDE_DIR};${PETSC_DIR}/${PETSC_ARCH}/include"
        )
    else()
        message(FATAL_ERROR "PETSc could not be found at ${_petsc_root}")
    endif()
endif()
