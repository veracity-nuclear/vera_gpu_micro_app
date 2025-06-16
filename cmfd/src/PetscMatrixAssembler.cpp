#include "PetscMatrixAssembler.hpp"

Mat SimpleMatrixAssembler::assemble() const
{
    Mat mat;
    PetscFunctionBeginUser;

    // Create the matrix (not allocated yet)
    MatCreate(PETSC_COMM_WORLD, &mat);

    // Set the matrix dimensions (just for compatibility checks))
    // The PETSC_DECIDEs are for sub matrices split across multiple MPI ranks.
    const PetscInt matSize = cmfdData.nCells * cmfdData.nGroups;
    MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, matSize, matSize);

    // Set the type of the matrix (etc.) based on PETSc CLI options.
    // Default is AIJ sparse matrix.
    MatSetFromOptions(mat);

    // TODO Figure out a way to change assembly type. Runtime vs compile time?
    // Should these just be different classes?
    // I think not using subviews is slightly faster. Not enough to matter right now
    // I think other assemblers will be more efficient than this one since we can use parallel algorithms
    // #define SIMPLE_MATRIX_USE_SUBVIEW
    #ifdef SIMPLE_MATRIX_USE_SUBVIEW
    for (PetscInt cellIdx = 0; cellIdx < cmfdData.nCells; ++cellIdx)
    {
        PetscScalar volume = cmfdData.volume(cellIdx);

        auto scatteringMat = Kokkos::subview(cmfdData.scatteringXS, Kokkos::ALL(), Kokkos::ALL(), cellIdx);
        for (PetscInt scatterFromIdx = 0; scatterFromIdx < cmfdData.nGroups; ++scatterFromIdx)
        {
            for (PetscInt scatterToIdx = 0; scatterToIdx < cmfdData.nGroups; ++scatterToIdx)
            {
                const PetscScalar value = -1 * scatteringMat(scatterToIdx, scatterFromIdx) * volume;
                if (value != 0.0)
                {
                    MatSetValue(mat, cellIdx * cmfdData.nGroups + scatterFromIdx, cellIdx * cmfdData.nGroups + scatterToIdx, value, ADD_VALUES);
                }
            }
        }


        auto transportMat = Kokkos::subview(cmfdData.transportXS, Kokkos::ALL(), cellIdx);
        for (PetscInt groupIdx = 0; groupIdx < cmfdData.nGroups; ++groupIdx)
        {
            const PetscScalar value = transportMat(groupIdx) * volume;
            if (value != 0.0)
            {
                const PetscInt diagIdx = cellIdx * cmfdData.nGroups + groupIdx;
                MatSetValue(mat, diagIdx, diagIdx, value, ADD_VALUES);
            }
        }
    }

    for (PetscInt surfaceIdx = 0; surfaceIdx < cmfdData.nSurfaces; ++surfaceIdx)
    {
        const PetscInt posCell = cmfdData.surf2Cell(surfaceIdx, 0);
        const PetscInt negCell = cmfdData.surf2Cell(surfaceIdx, 1);

        auto dHatMat = Kokkos::subview(cmfdData.dHat, Kokkos::ALL(), surfaceIdx);
        auto dTildeMat = Kokkos::subview(cmfdData.dTilde, Kokkos::ALL(), surfaceIdx);

        for (PetscInt groupIdx = 0; groupIdx < cmfdData.nGroups; ++groupIdx)
        {
            const PetscScalar dhat = dHatMat(groupIdx);
            const PetscScalar dtilde = dTildeMat(groupIdx);

            const PetscInt posCellMatIdx = (posCell) * cmfdData.nGroups + groupIdx;
            const PetscInt negCellMatIdx = (negCell) * cmfdData.nGroups + groupIdx;

            if (posCellMatIdx >= 0)
            {
                const PetscScalar value = -1 * dhat + dtilde;
                if (value != 0.0) {MatSetValue(mat, posCellMatIdx, posCellMatIdx, value, ADD_VALUES);}
            }
            if (negCellMatIdx >= 0)
            {
                const PetscScalar value = dhat + dtilde;
                if (value != 0.0) {MatSetValue(mat, negCellMatIdx, negCellMatIdx, value, ADD_VALUES);}
            }
            if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
            {
                const PetscScalar value1 = -1 * dhat - dtilde;
                if (value1 != 0.0) {MatSetValue(mat, posCellMatIdx, negCellMatIdx, value1, ADD_VALUES);}

                const PetscScalar value2 = dhat - dtilde;
                if (value2 != 0.0) {MatSetValue(mat, negCellMatIdx, posCellMatIdx, value2, ADD_VALUES);}
            }
        }
    }
    #else // Don't use subviews
    for (PetscInt scatterToIdx = 0; scatterToIdx < cmfdData.nGroups; ++scatterToIdx)
    {
        for (PetscInt scatterFromIdx = 0; scatterFromIdx < cmfdData.nGroups; ++scatterFromIdx)
        {
            for (PetscInt cellIdx = 0; cellIdx < cmfdData.nCells; ++cellIdx)
            {
                const PetscScalar value = -1 * cmfdData.scatteringXS(scatterToIdx, scatterFromIdx, cellIdx) * cmfdData.volume(cellIdx);
                if (value != 0.0)
                {
                    MatSetValue(mat, cellIdx * cmfdData.nGroups + scatterFromIdx, cellIdx * cmfdData.nGroups + scatterToIdx, value, ADD_VALUES);
                }
            }
        }
    }

    for (PetscInt groupIdx = 0; groupIdx < cmfdData.nGroups; ++groupIdx)
    {
        for (PetscInt cellIdx = 0; cellIdx < cmfdData.nCells; ++cellIdx)
        {
            // Could do removal XS if we were inserting instead of adding
            const PetscScalar value = cmfdData.transportXS(groupIdx, cellIdx) * cmfdData.volume(cellIdx);
            if (value != 0.0)
            {
                const PetscInt diagIdx = cellIdx * cmfdData.nGroups + groupIdx;
                MatSetValue(mat, diagIdx, diagIdx, value, ADD_VALUES);
            }
        }
    }

    for (PetscInt surfaceIdx = 0; surfaceIdx < cmfdData.nSurfaces; ++surfaceIdx)
    {
        const PetscInt posCell = cmfdData.surf2Cell(surfaceIdx, 0);
        const PetscInt negCell = cmfdData.surf2Cell(surfaceIdx, 1);

        for (PetscInt groupIdx = 0; groupIdx < cmfdData.nGroups; ++groupIdx)
        {
            const PetscScalar dhat = cmfdData.dHat(groupIdx, surfaceIdx);
            const PetscScalar dtilde = cmfdData.dTilde(groupIdx, surfaceIdx);

            const PetscInt posCellMatIdx = (posCell) * cmfdData.nGroups + groupIdx;
            const PetscInt negCellMatIdx = (negCell) * cmfdData.nGroups + groupIdx;

            if (posCellMatIdx >= 0)
            {
                const PetscScalar value = -1 * dhat + dtilde;
                if (value != 0.0) {MatSetValue(mat, posCellMatIdx, posCellMatIdx, value, ADD_VALUES);}
            }
            if (negCellMatIdx >= 0)
            {
                const PetscScalar value = dhat + dtilde;
                if (value != 0.0) {MatSetValue(mat, negCellMatIdx, negCellMatIdx, value, ADD_VALUES);}
            }
            if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
            {
                const PetscScalar value1 = -1 * dhat - dtilde;
                if (value1 != 0.0) {MatSetValue(mat, posCellMatIdx, negCellMatIdx, value1, ADD_VALUES);}

                const PetscScalar value2 = dhat - dtilde;
                if (value2 != 0.0) {MatSetValue(mat, negCellMatIdx, posCellMatIdx, value2, ADD_VALUES);}
            }
        }
    }
    #endif

    // Actually put the values into the matrix
    MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    return mat;
}

Mat COOMatrixAssembler::assemble() const
{
    Mat mat;
    PetscFunctionBeginUser;

    static constexpr size_t posDisplacement = 0;
    static constexpr size_t negPosDisplacement = 1;
    static constexpr size_t negNegDisplacement = 2;
    static constexpr size_t posNegDisplacement = 3;

    static constexpr PetscInt entriesPerSurf = 4; // Leakage in and out of pos and neg cells
    PetscInt maxNNZInRow = cmfdData.nGroups + entriesPerSurf * MAX_POS_SURF_PER_CELL;
    PetscInt matSize = cmfdData.nCells * cmfdData.nGroups;

    // // There are a lot of options here for splitting up the matrix into submatrices per mpi rank
    // //  (on the PETSc side), i.e., # of rows/cols per rank and number of zeros on and off the diagonal (per row).
    MatCreateAIJKokkos(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
        matSize, matSize,
        PETSC_DEFAULT, NULL,
        PETSC_DEFAULT, NULL,
        &mat);

    // Attempt. Could optimize with d_nnz based on scatter matrix layout (4th from last param)
    // MatCreateAIJKokkos(PETSC_COMM_WORLD, cmfdData.nGroups, cmfdData.nGroups,
    //     matSize, matSize, cmfdData.nGroups, NULL, 1, NULL, &mat);

    static constexpr int method = 2;
    // METHOD 1: Use Vectors with emplace back to avoid storing zeros
    if constexpr(method == 1)
    {
        std::vector<PetscInt> rowIndices, colIndices;
        std::vector<PetscScalar> values;
        rowIndices.reserve(maxNNZInRow * cmfdData.nCells);
        colIndices.reserve(maxNNZInRow * cmfdData.nCells);
        values.reserve(maxNNZInRow * cmfdData.nCells);

        for (PetscInt scatterToIdx = 0; scatterToIdx < cmfdData.nGroups; ++scatterToIdx)
        {
            for (PetscInt scatterFromIdx = 0; scatterFromIdx < cmfdData.nGroups; ++scatterFromIdx)
            {
                for (PetscInt cellIdx = 0; cellIdx < cmfdData.nCells; ++cellIdx)
                {
                    const PetscScalar value = -1 * cmfdData.scatteringXS(scatterToIdx, scatterFromIdx, cellIdx) * cmfdData.volume(cellIdx);
                    if (value != 0.0)
                    {
                        rowIndices.emplace_back(cellIdx * cmfdData.nGroups + scatterFromIdx);
                        colIndices.emplace_back(cellIdx * cmfdData.nGroups + scatterToIdx);
                        values.emplace_back(value);
                    }
                }
            }
        }

        for (PetscInt groupIdx = 0; groupIdx < cmfdData.nGroups; ++groupIdx)
        {
            for (PetscInt cellIdx = 0; cellIdx < cmfdData.nCells; ++cellIdx)
            {
                // Could do removal XS if we were inserting instead of adding
                const PetscScalar value = cmfdData.transportXS(groupIdx, cellIdx) * cmfdData.volume(cellIdx);
                if (value != 0.0)
                {
                    const PetscInt diagIdx = cellIdx * cmfdData.nGroups + groupIdx;
                    rowIndices.emplace_back(diagIdx);
                    colIndices.emplace_back(diagIdx);
                    values.emplace_back(value);
                }
            }
        }

        for (PetscInt surfaceIdx = 0; surfaceIdx < cmfdData.nSurfaces; ++surfaceIdx)
        {
            const PetscInt posCell = cmfdData.surf2Cell(surfaceIdx, 0);
            const PetscInt negCell = cmfdData.surf2Cell(surfaceIdx, 1);

            for (PetscInt groupIdx = 0; groupIdx < cmfdData.nGroups; ++groupIdx)
            {
                const PetscScalar dhat = cmfdData.dHat(groupIdx, surfaceIdx);
                const PetscScalar dtilde = cmfdData.dTilde(groupIdx, surfaceIdx);

                const PetscInt posCellMatIdx = (posCell) * cmfdData.nGroups + groupIdx;
                const PetscInt negCellMatIdx = (negCell) * cmfdData.nGroups + groupIdx;

                if (posCellMatIdx >= 0)
                {
                    const PetscScalar value = -1 * dhat + dtilde;
                    if (value != 0.0) {
                        rowIndices.emplace_back(posCellMatIdx);
                        colIndices.emplace_back(posCellMatIdx);
                        values.emplace_back(value);
                    }
                }
                if (negCellMatIdx >= 0)
                {
                    const PetscScalar value = dhat + dtilde;
                    if (value != 0.0) {
                        rowIndices.emplace_back(negCellMatIdx);
                        colIndices.emplace_back(negCellMatIdx);
                        values.emplace_back(value);
                    }
                }
                if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
                {
                    const PetscScalar value1 = -1 * dhat - dtilde;
                    if (value1 != 0.0) {
                        rowIndices.emplace_back(posCellMatIdx);
                        colIndices.emplace_back(negCellMatIdx);
                        values.emplace_back(value1);
                    }

                    const PetscScalar value2 = dhat - dtilde;
                    if (value2 != 0.0) {
                        rowIndices.emplace_back(negCellMatIdx);
                        colIndices.emplace_back(posCellMatIdx);
                        values.emplace_back(value2);
                    }
                }
            }
        }
        size_t numNonZero = values.size();


        MatSetPreallocationCOO(mat, numNonZero, rowIndices.data(), colIndices.data());
        MatSetValuesCOO(mat, values.data(), ADD_VALUES);
    }
    else if constexpr(method == 2) // METHOD 2:
    // Use Kokkos views somewhat naively (use teams, scratch pad, etc. to optimize)
    // Just making sure this works for now
    {
        // Don't want to access cmfdData in the lambda, so copy it to a reference
        auto& _cmfdData = cmfdData;

        // Assume each row has maxNNZInRow non-zero entries
        PetscInt maxNNZ = maxNNZInRow * matSize;

        // First maxNNZInRow are for the first row, next maxNNZInRow for the second row, etc.
        // First "row" is for the first cell first (scatter from) energy group, second "row" is for the first cell second energy group
        // In each "row", the first nGroups are for scattering from that group, and the last twelve are for leakage surfaces
        // There are three possible leakage positive leakage surfaces per cell and we have four entries per surface
        // That is, the matrix looks like this:
        //
        // Row0 (Cell 0, ScatterFrom 0): [ScatterTo0, ... ScatterToN, ++Surf0, -+Surf0, --Surf0, +-Surf0, ++Surf1, -+Surf1, --Surf1, +-Surf1, ++Surf2, -+Surf2, --Surf2, +-Surf2,
        // Row1 (Cell 0, ScatterFrom 1): [ScatterTo0, ... ScatterToN, ++Surf0, -+Surf0, --Surf0, +-Surf0, ++Surf1, -+Surf1, --Surf1, +-Surf1, ++Surf2, -+Surf2, --Surf2, +-Surf2,

        // Maybe these get assembled on the GPU, copied to the host for MatSetValuesCOO
        // Maybe this changes based on where the matrix is assembled, or does Kokkos handle that?
        Kokkos::View<PetscInt *, AssemblyMemorySpace> rowIndices("rowIndicesKokkos", maxNNZ);
        Kokkos::View<PetscInt *, AssemblyMemorySpace> colIndices("colIndicesKokkos", maxNNZ);
        Kokkos::View<PetscScalar *, AssemblyMemorySpace> values("valuesKokkos", maxNNZ);

        {
            Kokkos::parallel_for("COOMatrixAssembler2: ScatterXS", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<3>>({0, 0, 0}, {cmfdData.nGroups, cmfdData.nGroups, cmfdData.nCells}),
                KOKKOS_LAMBDA(const PetscInt scatterToIdx, const PetscInt scatterFromIdx, const PetscInt cellIdx)
            {
                // Don't need diagonal entries here because we use removal xs
                // Else, we would have to have another entry for the diagonal
                if (scatterToIdx != scatterFromIdx)
                {
                    const size_t rowIdxInMat = cellIdx * _cmfdData.nGroups + scatterFromIdx;
                    const size_t locationIn1D = rowIdxInMat * maxNNZInRow + scatterToIdx;
                    rowIndices(locationIn1D) = rowIdxInMat;
                    colIndices(locationIn1D) = cellIdx * _cmfdData.nGroups + scatterToIdx;
                    values(locationIn1D) = -1 * _cmfdData.scatteringXS(scatterToIdx, scatterFromIdx, cellIdx) * _cmfdData.volume(cellIdx);
                }
            });

            Kokkos::parallel_for("COOMatrixAssembler2: RemovalXS", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<2>>({0, 0}, {cmfdData.nGroups, cmfdData.nCells}),
                KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt cellIdx)
            {
                // groupIdx is both from and to since removal includes self scattering
                const size_t rowIdxInMat = cellIdx * _cmfdData.nGroups + groupIdx;
                const size_t locationIn1D = rowIdxInMat * maxNNZInRow + groupIdx;
                rowIndices(locationIn1D) = rowIdxInMat;
                colIndices(locationIn1D) = rowIdxInMat; // diagonal
                values(locationIn1D) = _cmfdData.removalXS(groupIdx, cellIdx) * _cmfdData.volume(cellIdx);
            });

            // TODO: This isn't tightly nested and could be optimized
            Kokkos::parallel_for("COOMatrixAssembler2: Leakage", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<3>>({0, 0, 0}, {cmfdData.nGroups, cmfdData.nCells, MAX_POS_SURF_PER_CELL}),
                KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt posCellIdx, const PetscInt posSurfPosition)
            {
                // We are only on the diagonal so groupIdx could be from or to, but we use it to find the "row" (scatter from)
                // posSurfPosition is 0, 1, or 2
                const PetscInt posSurfIdx = _cmfdData.cell2PosSurf(posCellIdx, posSurfPosition);
                if (posSurfIdx >= 0) // -1 is invalid surface
                {
                    const PetscScalar dhat = _cmfdData.dHat(groupIdx, posSurfIdx);
                    const PetscScalar dtilde = _cmfdData.dTilde(groupIdx, posSurfIdx);

                    const PetscInt posCellMatIdx = (posCellIdx) * _cmfdData.nGroups + groupIdx;
                    const size_t locationStartIn1D = posCellMatIdx * maxNNZInRow + _cmfdData.nGroups + posSurfPosition * entriesPerSurf;

                    const PetscInt negCellIdx = _cmfdData.surf2Cell(posSurfIdx, 1);
                    const PetscInt negCellMatIdx = (negCellIdx) * _cmfdData.nGroups + groupIdx;

                    rowIndices(locationStartIn1D + posDisplacement) = posCellMatIdx;
                    colIndices(locationStartIn1D + posDisplacement) = posCellMatIdx;
                    values(locationStartIn1D + posDisplacement) = -1 * dhat + dtilde;

                    if (negCellIdx >= 0)
                    {
                        rowIndices(locationStartIn1D + negNegDisplacement) = negCellMatIdx;
                        colIndices(locationStartIn1D + negNegDisplacement) = negCellMatIdx;
                        values(locationStartIn1D + negNegDisplacement) = dhat + dtilde;

                        rowIndices(locationStartIn1D + negPosDisplacement) = negCellMatIdx;
                        colIndices(locationStartIn1D + negPosDisplacement) = posCellMatIdx;
                        values(locationStartIn1D + negPosDisplacement) = dhat - dtilde;

                        rowIndices(locationStartIn1D + posNegDisplacement) = posCellMatIdx;
                        colIndices(locationStartIn1D + posNegDisplacement) = negCellMatIdx;
                        values(locationStartIn1D + posNegDisplacement) = -1 * dhat - dtilde;
                    }

                }
            });
        }

        MatSetPreallocationCOO(mat, maxNNZ, rowIndices.data(), colIndices.data());
        MatSetValuesCOO(mat, values.data(), ADD_VALUES);
    }
    return mat;
}