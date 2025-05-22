#include <petscsys.h>
#include <gtest/gtest.h>

TEST(PETScTest, InitializeAndCheckMPI) {
    PetscInitialize(NULL, NULL, NULL, NULL);

    PetscPrintf(PETSC_COMM_WORLD, "Hello from PETSc!\n");

    PetscFinalize();
}
