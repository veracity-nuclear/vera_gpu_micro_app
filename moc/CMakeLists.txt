add_library(moc_lib
    src/main.cpp
    src/serial_moc.cpp
    src/c5g7_library.cpp
)

target_link_libraries(moc_lib
    HighFive::HighFive
    Kokkos::kokkos
)

add_executable(micro_moc
    src/main.cpp
)
target_link_libraries(micro_moc moc_lib)

# Copy the data directory to the build location
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --- Tests ---
enable_testing()
add_executable(moc_tests
    tests/test_main.cpp
)

target_link_libraries(moc_tests
    moc_lib
    HighFive::HighFive
    Kokkos::kokkos
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME moc_tests COMMAND moc_tests)
