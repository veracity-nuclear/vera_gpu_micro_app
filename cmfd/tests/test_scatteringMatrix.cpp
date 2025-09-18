#include "test_scatteringMatrix.hpp"

INSTANTIATE_TEST_SUITE_P(
    TestBuildScatteringMatrix,
    ScatteringMatrixTest,
    ::testing::Values(
        "data/pin_7g_16a_3p_serial.h5",
        "data/2x2_7g_16a_3p_serial.h5"
    ));