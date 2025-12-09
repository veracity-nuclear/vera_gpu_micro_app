cmake_minimum_required(VERSION 3.16)
project(VERA_GPU_MicroApps C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Allow setting default search paths for find_package()
if(NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "$ENV{HDF5_ROOT};$ENV{HIGHFIVE_ROOT};$ENV{KOKKOS_ROOT};$ENV{PETSC_DIR}" CACHE STRING "Default CMAKE_PREFIX_PATH")
endif()

# Find Kokkos (user-installed)
find_package(Kokkos REQUIRED)

# Find PETSc (via custom module or manual)
find_package(PETSc REQUIRED)

# Find HighFive (via custom module or manual)
find_package(HighFive REQUIRED)

# GTest via FetchContent (if not system-installed)
find_package(googletest)
IF (${googletest_FOUND})
    MESSAGE(STATUS "googletest installation found: ${GTEST_LIBRARY}")
ELSE()
    MESSAGE(STATUS "googletest not found; defaulting to FetchContent")
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
ENDIF()

enable_testing()

# Add utils before other micro-apps
add_subdirectory(utils)

# Add micro-apps
add_subdirectory(cmfd)
add_subdirectory(moc)
add_subdirectory(conduction)
add_subdirectory(subchannel)
