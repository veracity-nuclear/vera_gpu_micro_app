#include "test_DavidsonEigenSolver.hpp"

static const std::vector<std::string> testFiles = {
    "data/pin_7g_16a_3p_serial.h5",
    "data/7x7_7g_16a_3p_serial.h5",
    "data/small_parallel.h5"
};

INSTANTIATE_TEST_SUITE_P(
    TestDavidsonSimpleLight,
    DavidsonEigenSolverTest,
    ::testing::ValuesIn(createParams<SimpleMatrixAssembler>(testFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestDavidsonCOOLight,
    DavidsonEigenSolverTest,
    ::testing::ValuesIn(createParams<COOMatrixAssembler>(testFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestDavidsonCSRLight,
    DavidsonEigenSolverTest,
    ::testing::ValuesIn(createParams<CSRMatrixAssembler>(testFiles))
);
