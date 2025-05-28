add_library(cmfd_lib
    src/main.cpp
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
set(TEST_EXECUTABLES
    test_main
    test_PETSc_ex3k.kokkos
    test_preliminary
)
foreach(test ${TEST_EXECUTABLES})
    set(test_exe_name "CMFD-${test}")
    add_executable(${test_exe_name} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test}.cpp)
    target_link_libraries(${test_exe_name}
        cmfd_lib
        HighFive::HighFive
        Kokkos::kokkos
        GTest::gtest
        GTest::gtest_main
    )
    target_include_directories(${test_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ${test_exe_name} COMMAND ${test_exe_name})
    message(STATUS "Added test: ${test}")
endforeach()