add_library(subchannel_lib
    src/geometry.cpp
    src/solver.cpp
    src/th.cpp
)

target_link_libraries(subchannel_lib
    HighFive::HighFive
    Kokkos::kokkos
    PETSc::PETSc
    utils
)

add_executable(micro_subchannel
    src/main.cpp
)
target_link_libraries(micro_subchannel subchannel_lib)

# Copy the data directory to the build location
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --- Tests ---
enable_testing()
set(TEST_EXECUTABLES
    test_geometry
    test_ants_1x1_serial
    test_ants_3x3_omp
    test_ants_3x3_serial
    test_ants_7x7_omp
    test_ants_7x7_serial
    test_ants_minicore_serial
    test_ants_minicore_omp
    test_ants_smr_serial
    test_ants_smr_omp
)
set(TEST_LABELS
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "BASIC\;CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "CONTINUOUS\;HEAVY"
    "HEAVY"
    "HEAVY"
)
list(LENGTH TEST_EXECUTABLES list_length)
math(EXPR last_index "${list_length} - 1")
foreach(i RANGE 0 ${last_index})
    list(GET TEST_EXECUTABLES ${i} test)
    list(GET TEST_LABELS ${i} labels)
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test}.cpp)
    target_link_libraries(${test}
        subchannel_lib
        GTest::gtest
    )
    target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${test} PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS "Subchannel;${labels}")
    message(STATUS "Added subchannel micro-app test: ${test}")
endforeach()
