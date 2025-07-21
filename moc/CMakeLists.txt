add_library(moc_lib
    src/main.cpp
    src/eigen_solver.cpp
    src/serial_moc.cpp
    src/kokkos_moc.cpp
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
include(../cmake/TestFunctions.cmake)

set(TEST_EXECUTABLES
    test_exp_table
    test_serial_moc_1region_1angle_single
    test_serial_moc_4region_1angle_single
    test_serial_moc_1region_16angle_single
    test_serial_moc_1region_1angle_double
    test_serial_moc_4region_1angle_double
    test_serial_moc_1region_16angle_double
    test_mixed_bcs
    test_serial_moc_pin_single
    test_serial_moc_pin_double
    test_kokkos_moc_pin_omp4_single
    test_kokkos_moc_pin_serial_single
    test_kokkos_moc_pin_omp4_double
    test_kokkos_moc_pin_serial_double
    test_serial_moc_7x7_single
    test_serial_moc_7x7_double
    test_kokkos_moc_7x7_serial_single
    test_kokkos_moc_7x7_omp4_single
    test_kokkos_moc_7x7_serial_double
    test_kokkos_moc_7x7_omp4_double
    test_serial_moc_mini-core_single
    test_serial_moc_mini-core_double
)
SET(TEST_LABELS
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "HEAVY"
    "HEAVY"
    "HEAVY"
    "HEAVY"
    "HEAVY"
    "HEAVY"
    "HEAVY"
    "HEAVY"
)

list(LENGTH TEST_EXECUTABLES list_length)
math(EXPR last_index "${list_length} - 1")
foreach(i RANGE 0 ${last_index})
    list(GET TEST_EXECUTABLES ${i} test)
    list(GET TEST_LABELS ${i} labels)
    create_labeled_test(${test} ${CMAKE_CURRENT_SOURCE_DIR}/tests "moc_lib;HighFive::HighFive;Kokkos::kokkos;GTest::gtest" ${CMAKE_CURRENT_SOURCE_DIR}/src "${labels};MOC")
    set_tests_properties(${test} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

