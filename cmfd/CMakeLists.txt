add_library(cmfd_lib
    src/main.cpp
    src/PetscMatrixAssembler.cpp
    src/PetscEigenSolver.cpp
)

target_link_libraries(cmfd_lib
    HighFive::HighFive
    Kokkos::kokkos
    PETSc::PETSc
    utils
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
    test_PetscMatrixAssemblerLight
    test_PetscMatrixAssemblerHeavy
    test_PetscEigenSolverLight
    test_PetscEigenSolverHeavy
)
set(CMFD_TEST_SUITE_LABELS
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "HEAVY"
    "CONTINUOUS\;HEAVY"
    "HEAVY"
)

set(COMMON_TEST_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp)

list(LENGTH CMFD_TEST_SUITE_FILES list_length)
math(EXPR last_index "${list_length} - 1")
foreach(i RANGE 0 ${last_index})
    list(GET CMFD_TEST_SUITE_FILES ${i} test_suite_basename)
    set(test_exe_name "CMFD-${test_suite_basename}")
    list(GET CMFD_TEST_SUITE_LABELS ${i} labels)

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
    set_tests_properties(${test_exe_name} PROPERTIES LABELS "CMFD;${labels}")
    message(STATUS "Added test: ${test_exe_name} (using common TestMain.cpp) with labels: ${labels}")
endforeach()
