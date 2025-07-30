#include "test_PetscEigenSolver.hpp"

static const std::vector<std::string> lightTestFiles = {
    "data/pin_7g_16a_3p_serial.h5",
    "data/7x7_7g_16a_3p_serial.h5"
};

INSTANTIATE_TEST_SUITE_P(
    TestEigenSimpleLight,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<SimpleMatrixAssembler>(lightTestFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestEigenCOOLight,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<COOMatrixAssembler>(lightTestFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestEigenCSRLight,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<CSRMatrixAssembler>(lightTestFiles))
);