#include "PetscMatrixAssembler.hpp"

Mat SimpleMatrixAssembler::assemble() const
{
    Mat mat;
    PetscFunctionBeginUser;

    MatCreate(PETSC_COMM_WORLD, &mat);
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, cmfdData.nCells, cmfdData.nCells);
    MatSetFromOptions(mat);

    return mat;
}