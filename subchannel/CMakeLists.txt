# ANTS Subchannel Solver Library
add_library(ants_subchannel STATIC
    src/Constants.hpp
    src/Properties.hpp
    src/Properties.cpp
    src/SubchannelData.hpp
    src/SubchannelData.cpp
    src/ThermalHydraulics.hpp
    src/ThermalHydraulics.cpp
    src/Solver.hpp
    src/Solver.cpp
)

target_include_directories(ants_subchannel PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(ants_subchannel PUBLIC cxx_std_17)

# ANTS 3x3 Verification Executable  
add_executable(ants_3x3_verification
    tests/verification_3x3.cpp
)

target_link_libraries(ants_3x3_verification 
    ants_subchannel
)

target_compile_features(ants_3x3_verification PUBLIC cxx_std_17)

# --- Tests ---
enable_testing()

# Keep original test executable
add_executable(subchannel_tests
    tests/test_main.cpp
)

target_link_libraries(subchannel_tests
    ants_subchannel
    GTest::gtest
)

add_test(NAME subchannel_tests COMMAND subchannel_tests)
set_tests_properties(subchannel_tests PROPERTIES LABELS "Subchannel;BASIC;CONTINUOUS;HEAVY")

# Add ANTS verification test
add_test(NAME ants_3x3_verification COMMAND ants_3x3_verification)
set_tests_properties(ants_3x3_verification PROPERTIES LABELS "Subchannel;VERIFICATION;CONTINUOUS")
