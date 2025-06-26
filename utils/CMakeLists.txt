add_library(utils
    src/argument_parser.cpp
)

target_include_directories(utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(utils
    HighFive::HighFive
)

# --- Tests ---
enable_testing()
set(TEST_EXECUTABLES
    test_argument_parser
    test_hdf5_utils
)
foreach(test ${TEST_EXECUTABLES})
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test}.cpp)
    target_link_libraries(${test}
        utils
        GTest::gtest
    )
    target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${test} PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../conduction/data")
    add_test(NAME ${test} COMMAND ${test})
    set_tests_properties(${test} PROPERTIES LABELS "Utils;BASIC;CONTINUOUS;HEAVY")
    message(STATUS "Added test: ${test}")
endforeach()
