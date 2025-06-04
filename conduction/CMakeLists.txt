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
add_executable(conduction_tests
    tests/test_main.cpp
    tests/test_highfive.cpp
    tests/test_kokkos.cpp
    tests/test_petsc.cpp
)

target_link_libraries(conduction_tests
    conduction_lib
    HighFive::HighFive
    Kokkos::kokkos
    PETSc::PETSc
    GTest::gtest
)

add_test(NAME conduction_tests COMMAND conduction_tests)
