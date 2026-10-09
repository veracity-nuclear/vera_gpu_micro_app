#include "test_DavidsonEigenSolver.hpp"

static const std::vector<std::string> testFiles = {
    "data/mini-core_7g_16a_3p_serial.h5",
};

INSTANTIATE_TEST_SUITE_P(
    TestDavidsonSimpleHeavy,
    DavidsonEigenSolverTest,
    ::testing::ValuesIn(createParams<SimpleMatrixAssembler>(testFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestDavidsonCOOHeavy,
    DavidsonEigenSolverTest,
    ::testing::ValuesIn(createParams<COOMatrixAssembler>(testFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestDavidsonCSRHeavy,
    DavidsonEigenSolverTest,
    ::testing::ValuesIn(createParams<CSRMatrixAssembler>(testFiles))
);
