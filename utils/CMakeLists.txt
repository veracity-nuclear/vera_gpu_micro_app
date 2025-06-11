add_library(utils
    src/argument_parser.cpp
)

target_include_directories(utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(utils)

# --- Tests ---
enable_testing()
set(TEST_EXECUTABLES
    test_argument_parser
)
foreach(test ${TEST_EXECUTABLES})
    add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/tests/${test}.cpp)
    target_link_libraries(${test}
        utils
        GTest::gtest
    )
    target_include_directories(${test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME ${test} COMMAND ${test})
    message(STATUS "Added test: ${test}")
endforeach()