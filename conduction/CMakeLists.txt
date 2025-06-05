add_library(conduction_lib
    src/main.cpp
)

target_link_libraries(conduction_lib
    HighFive::HighFive
    Kokkos::kokkos
    PETSc::PETSc
)

add_executable(conduction_exec
    src/main.cpp
)
target_link_libraries(conduction_exec conduction_lib)

# --- Tests ---
enable_testing()
set(TEST_EXECUTABLES
    test_cylinder_node
    test_cylindrical_solver
    test_hdf5_utils
)
foreach(test ${TEST_EXECUTABLES})
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test}.cpp)
    target_link_libraries(${test}
        conduction_lib
        GTest::gtest
        GTest::gtest_main
    )
    target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${test} PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_test(NAME conduction_${test} COMMAND ${test})
    message(STATUS "Added conduction micro-app test: ${test}")
endforeach()
