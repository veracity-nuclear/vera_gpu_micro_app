#include "PetscMatrixAssembler.hpp"

Mat SimpleMatrixAssembler::assemble() const
{
    Mat mat;
    PetscFunctionBeginUser;

    MatCreate(PETSC_COMM_WORLD, &mat);
    // MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, m_nCells, m_nCells);
    // MatSetFromOptions(mat);

    return mat;
}