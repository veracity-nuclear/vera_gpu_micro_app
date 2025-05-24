#include <petscsys.h>
#include <petsc.h>
#include <gtest/gtest.h>

TEST(PETScTest, BasicTest) {
    PetscPrintf(PETSC_COMM_WORLD, "Hello from PETSc!\n");

    // Initialize a PETSc vector
    Vec vec;
    VecCreate(PETSC_COMM_WORLD, &vec);
    VecSetSizes(vec, PETSC_DECIDE, 10);
    VecSetFromOptions(vec);
    VecSet(vec, 10.0);
    VecAssemblyBegin(vec);
    VecAssemblyEnd(vec);
    VecView(vec, PETSC_VIEWER_STDOUT_WORLD);
    VecDestroy(&vec);
}
