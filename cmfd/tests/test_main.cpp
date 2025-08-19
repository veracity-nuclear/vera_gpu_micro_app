#include "gtest/gtest.h"
#include "PetscKokkosTestEnvironment.hpp"

TEST(IntegrationMain, RunsOnSample) {
  // Test if the main executable runs with a sample input file
  int code = std::system("./cmfd_exec data/pin_7g_16a_3p_serial.h5");
  ASSERT_EQ(code, 0);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new PetscKokkosTestEnvironment(argc, argv));
  return RUN_ALL_TESTS();
}