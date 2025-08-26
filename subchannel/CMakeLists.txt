add_library(subchannel_lib
    src/geometry.cpp
    src/materials.cpp
    src/solver.cpp
    src/th.cpp
)

target_link_libraries(subchannel_lib
    HighFive::HighFive
    Kokkos::kokkos
    PETSc::PETSc
    utils
)

# --- Tests ---
enable_testing()
set(TEST_EXECUTABLES
    test_single_channel
)
set(TEST_LABELS
    "BASIC\;CONTINUOUS\;HEAVY"
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
