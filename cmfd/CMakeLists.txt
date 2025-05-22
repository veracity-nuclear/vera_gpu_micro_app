add_library(cmfd_lib
    src/main.cpp
)

target_link_libraries(cmfd_lib
    Kokkos::kokkos
)

add_executable(cmfd_exec
    src/main.cpp
)
target_link_libraries(cmfd_exec cmfd_lib)

# --- Tests ---
enable_testing()
add_executable(cmfd_tests
    tests/test_main.cpp
)

target_link_libraries(cmfd_tests
    cmfd_lib
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME cmfd_tests COMMAND cmfd_tests)
