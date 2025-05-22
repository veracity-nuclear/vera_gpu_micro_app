add_library(conduction_lib
    src/main.cpp
)

target_link_libraries(conduction_lib
    Kokkos::kokkos
)

add_executable(conduction_exec
    src/main.cpp
)
target_link_libraries(conduction_exec conduction_lib)

# --- Tests ---
enable_testing()
add_executable(conduction_tests
    tests/test_main.cpp
)

target_link_libraries(conduction_tests
    conduction_lib
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME conduction_tests COMMAND conduction_tests)
