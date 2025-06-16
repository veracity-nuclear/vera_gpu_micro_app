#include "gtest/gtest.h"
#include "PetscKokkosTestEnvironment.hpp"

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new PetscKokkosTestEnvironment(argc, argv));
  return RUN_ALL_TESTS();
}