add_library(moc_lib
    src/main.cpp
    src/eigen_solver.cpp
    src/serial_moc.cpp
    src/c5g7_library.cpp
    src/exp_table.cpp
)

target_link_libraries(moc_lib
    HighFive::HighFive
    Kokkos::kokkos
    utils
)

add_executable(micro_moc
    src/main.cpp
)
target_link_libraries(micro_moc moc_lib)

# Copy the data directory to the build location
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --- Tests ---
enable_testing()
set(TEST_EXECUTABLES
    test_serial_moc_1region_1angle
    test_serial_moc_1region_16angle
    test_serial_moc_4region_1angle
    test_serial_moc_pin
    test_serial_moc_7x7
    heavy_test_serial_moc_mini-core
)
foreach(test ${TEST_EXECUTABLES})
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test}.cpp)
    target_link_libraries(${test}
        moc_lib
        HighFive::HighFive
        Kokkos::kokkos
        GTest::gtest
        GTest::gtest_main
    )
    target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ${test} COMMAND ${test})
    message(STATUS "Added test: ${test}")
endforeach()
