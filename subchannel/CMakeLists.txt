add_library(subchannel_lib
    src/main.cpp
)

target_link_libraries(subchannel_lib
    Kokkos::kokkos
    PETSc::PETSc
)

add_executable(subchannel_exec
    src/main.cpp
)
target_link_libraries(subchannel_exec subchannel_lib)

# --- Tests ---
enable_testing()
add_executable(subchannel_tests
    tests/test_main.cpp
)

target_link_libraries(subchannel_tests
    subchannel_lib
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME subchannel_tests COMMAND subchannel_tests)
