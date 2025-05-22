add_library(moc_lib
    src/main.cpp
)

target_link_libraries(moc_lib
    Kokkos::kokkos
    PETSc::PETSc
)

add_executable(moc_exec
    src/main.cpp
)
target_link_libraries(moc_exec moc_lib)

# --- Tests ---
enable_testing()
add_executable(moc_tests
    tests/test_main.cpp
)

target_link_libraries(moc_tests
    moc_lib
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME moc_tests COMMAND moc_tests)
