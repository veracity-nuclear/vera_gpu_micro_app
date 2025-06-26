#include "test_PetscMatrixAssembler.hpp"

INSTANTIATE_TEST_SUITE_P(
    TestAssemblyLight,
    PetscMatrixAssemblerTest,
    ::testing::Values(
      "data/pin_7g_16a_3p_serial.h5",
      "data/7x7_7g_16a_3p_serial.h5"));