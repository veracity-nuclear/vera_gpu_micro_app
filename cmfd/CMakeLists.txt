add_library(cmfd_lib
    src/main.cpp
    src/PetscMatrixAssembler.cpp
    src/PetscEigenSolver.cpp
)

target_link_libraries(cmfd_lib
    HighFive::HighFive
    Kokkos::kokkos
    PETSc::PETSc
)

add_executable(cmfd_exec
    src/main.cpp
)
target_link_libraries(cmfd_exec cmfd_lib)

# Copy the data directory to the build location
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --- Tests ---
enable_testing()

# Define the test environment implementation once
set(TEST_ENV_IMPL ${CMAKE_CURRENT_SOURCE_DIR}/tests/PetscKokkosTestEnvironment.cpp)

set(CMFD_TEST_SUITE_FILES
    test_PETSc_ex3k.kokkos
    test_preliminary
    test_CMFDData
    test_PetscMatrixAssembler
    test_PetscEigenSolver
)

set(COMMON_TEST_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp)

foreach(test_suite_basename ${CMFD_TEST_SUITE_FILES})
    set(test_exe_name "CMFD-${test_suite_basename}")
    set(test_suite_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test_suite_basename}.cpp)

    add_executable(${test_exe_name}
        ${COMMON_TEST_MAIN_SRC}       # Common main for all tests
        ${TEST_ENV_IMPL}              # Add the implementation file
        ${test_suite_src}             # The specific test suite file
    )

    target_link_libraries(${test_exe_name}
        cmfd_lib
        GTest::gtest
    )

    target_include_directories(${test_exe_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src    # For headers from your src directory
        ${CMAKE_CURRENT_SOURCE_DIR}/tests  # For PetscKokkosTestEnvironment.hpp
    )

    add_test(NAME ${test_exe_name} COMMAND ${test_exe_name})
    message(STATUS "Added test: ${test_exe_name} (using common TestMain.cpp)")
endforeach()