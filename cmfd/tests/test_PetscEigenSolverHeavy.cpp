#include "test_PetscEigenSolver.hpp"

static const std::vector<std::string> heavyTestFiles = {
    "data/mini-core_7g_16a_3p_serial.h5",
};

INSTANTIATE_TEST_SUITE_P(
    TestEigenSimpleHeavy,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<SimpleMatrixAssembler>(heavyTestFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestEigenCOOHeavy,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<COOMatrixAssembler>(heavyTestFiles))
);