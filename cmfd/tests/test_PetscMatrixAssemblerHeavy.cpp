#include "test_PetscMatrixAssembler.hpp"

INSTANTIATE_TEST_SUITE_P(
    TestAssemblyHeavy,
    PetscMatrixAssemblerTest,
    ::testing::Values(
      "data/mini-core_7g_16a_3p_serial.h5"));