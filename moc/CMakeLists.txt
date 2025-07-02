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
    test_serial_moc_1region_1angle
    test_serial_moc_4region_1angle
    test_serial_moc_1region_16angle
    test_serial_moc_pin
    test_kokkos_moc_pin_omp4
    test_kokkos_moc_pin_serial
    test_serial_moc_7x7
    test_kokkos_moc_7x7_serial
    test_kokkos_moc_7x7_omp4
    heavy_test_serial_moc_mini-core
)
SET(TEST_LABELS
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
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

