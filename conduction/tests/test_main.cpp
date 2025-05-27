#include <petscsys.h>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

TEST(BasicTest, AssertTrue) {
    EXPECT_TRUE(true);
}

TEST(BasicTest, AssertFalse) {
    EXPECT_FALSE(false);
}

TEST(BasicTest, AssertEqual) {
    EXPECT_EQ(1, 1);
}

int main(int argc, char **argv) {

    // Initialize PETSc
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    ::testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();

    // Finalize Kokkos
    Kokkos::finalize();

    // Finalize PETSc
    PetscFinalize();

    return test_result;
}
