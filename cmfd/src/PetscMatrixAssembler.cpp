#include "PetscMatrixAssembler.hpp"

inline bool isNonZero(const PetscScalar& value)
{
    return std::abs(value) > PETSC_MACHINE_EPSILON;
}

PetscErrorCode SimpleMatrixAssembler::_assembleM()
{
    PetscFunctionBeginUser;

    // Create the matrix (not allocated yet)
    MatCreate(PETSC_COMM_WORLD, &MMat);

    // Set the matrix dimensions (just for compatibility checks))
    // The PETSC_DECIDEs are for sub matrices split across multiple MPI ranks.
    const PetscInt matSize = cmfdData.nCells * cmfdData.nGroups;
    PetscCall(MatSetSizes(MMat, PETSC_DECIDE, PETSC_DECIDE, matSize, matSize));

    // Set the type of the matrix (etc.) based on PETSc CLI options.
    // Default is AIJ sparse matrix.
    PetscCall(MatSetFromOptions(MMat));

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
                if (isNonZero(value))
                {
                    PetscCall(MatSetValue(MMat, cellIdx * cmfdData.nGroups + scatterFromIdx, cellIdx * cmfdData.nGroups + scatterToIdx, value, ADD_VALUES));
                }
            }
        }


        auto transportMat = Kokkos::subview(cmfdData.transportXS, Kokkos::ALL(), cellIdx);
        for (PetscInt groupIdx = 0; groupIdx < cmfdData.nGroups; ++groupIdx)
        {
            const PetscScalar value = transportMat(groupIdx) * volume;
            if (isNonZero(value))
            {
                const PetscInt diagIdx = cellIdx * cmfdData.nGroups + groupIdx;
                PetscCall(MatSetValue(MMat, diagIdx, diagIdx, value, ADD_VALUES));
            }
            {
                const PetscInt diagIdx = cellIdx * cmfdData.nGroups + groupIdx;
                PetscCall(MatSetValue(MMat, diagIdx, diagIdx, value, ADD_VALUES));
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
                if (isNonZero(value)) {PetscCall(MatSetValue(MMat, posCellMatIdx, posCellMatIdx, value, ADD_VALUES));}
            }
            if (negCellMatIdx >= 0)
            {
                const PetscScalar value = dhat + dtilde;
                if (isNonZero(value)) {PetscCall(MatSetValue(MMat, negCellMatIdx, negCellMatIdx, value, ADD_VALUES));}
            }
            if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
            {
                const PetscScalar value1 = -1 * dhat - dtilde;
                if (isNonZero(value1)) {PetscCall(MatSetValue(MMat, posCellMatIdx, negCellMatIdx, value1, ADD_VALUES));}

                const PetscScalar value2 = dhat - dtilde;
                if (isNonZero(value2)) {PetscCall(MatSetValue(MMat, negCellMatIdx, posCellMatIdx, value2, ADD_VALUES));}
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
                if (isNonZero(value))
                {
                    PetscCall(MatSetValue(MMat, cellIdx * cmfdData.nGroups + scatterFromIdx, cellIdx * cmfdData.nGroups + scatterToIdx, value, ADD_VALUES));
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
            if (isNonZero(value))
            {
                const PetscInt diagIdx = cellIdx * cmfdData.nGroups + groupIdx;
                PetscCall(MatSetValue(MMat, diagIdx, diagIdx, value, ADD_VALUES));
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
                if (isNonZero(value)) {PetscCall(MatSetValue(MMat, posCellMatIdx, posCellMatIdx, value, ADD_VALUES));}
            }
            if (negCellMatIdx >= 0)
            {
                const PetscScalar value = dhat + dtilde;
                if (isNonZero(value)) {PetscCall(MatSetValue(MMat, negCellMatIdx, negCellMatIdx, value, ADD_VALUES));}
            }
            if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
            {
                const PetscScalar value1 = -1 * dhat - dtilde;
                if (isNonZero(value1)) {PetscCall(MatSetValue(MMat, posCellMatIdx, negCellMatIdx, value1, ADD_VALUES));}

                const PetscScalar value2 = dhat - dtilde;
                if (isNonZero(value2)) {PetscCall(MatSetValue(MMat, negCellMatIdx, posCellMatIdx, value2, ADD_VALUES));}
            }
        }
    }
    #endif

    // Actually put the values into the matrix
    PetscCall(MatAssemblyBegin(MMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(MMat, MAT_FINAL_ASSEMBLY));

    return PETSC_SUCCESS;
}

PetscErrorCode SimpleMatrixAssembler::_assembleFission(const FluxView& flux)
{
    std::vector<PetscInt> vecIndices;
    std::vector<PetscScalar> vecValues;
    vecIndices.reserve(nRows);
    vecValues.reserve(nRows);

    PetscFunctionBeginUser;

    for (PetscInt cellIdx = 0; cellIdx < cmfdData.nCells; ++cellIdx)
    {
        PetscScalar localFissionRate = 0.0;
        for (PetscInt fromGroupIdx = 0; fromGroupIdx < cmfdData.nGroups; ++fromGroupIdx)
        {
            // if FluxView is 2D
            // localFissionRate += cmfdData.nuFissionXS(fromGroupIdx, cellIdx) * flux(fromGroupIdx, cellIdx);

            // if FluxView is 1D
            localFissionRate += cmfdData.nuFissionXS(fromGroupIdx, cellIdx) * flux(cellIdx * cmfdData.nGroups + fromGroupIdx);
        }

        const PetscScalar localFissionRateVolume = localFissionRate * cmfdData.volume(cellIdx);

        for (PetscInt toGroupIdx = 0; toGroupIdx < cmfdData.nGroups; ++toGroupIdx)
        {
            const PetscScalar neutronSource = cmfdData.chi(toGroupIdx, cellIdx) * localFissionRateVolume;
            if (isNonZero(neutronSource))
            {
                const PetscInt vecIdx = cellIdx * cmfdData.nGroups + toGroupIdx;
                vecIndices.push_back(vecIdx);
                vecValues.push_back(neutronSource);
            }
        }
    }

    // Actually put the values into the vector
    PetscCall(VecSetValues(fissionVec, vecIndices.size(), vecIndices.data(), vecValues.data(), INSERT_VALUES));
    PetscCall(VecAssemblyBegin(fissionVec));
    PetscCall(VecAssemblyEnd(fissionVec));

    return PETSC_SUCCESS;
}

PetscErrorCode COOMatrixAssembler::_assembleM()
{
    /*
    The shape of a two cell three group matrix is
        X X X | X 0 0                 |
        X X X...0 X 0            + +  |  + -
        X X X | 0 0 X                 |
        - : - | - : -           - - - | - - -
        X 0 0 | X X X                 |
        0 X 0...X X X            - +  |  - -
        0 0 X | X X X                 |
    where the upper left submatrix is the scattering matrix for the first cell,
    and the lower right submatrix is the scattering matrix for the second cell.
    With more cells, each block on the diagonal is a scattering submatrix for a cell.
    For the leakage between any two cells, we fill the diagonal of the submatrix blocks
    that are line up with the corresponding cells' scattering submatrices. With more cells,
    there may be space between these submatrices.

    Leakage terms are added to to the diagonal of all four submatrices. We call the submatrix
    of the positive cell that is, the cell on the positive side of the surface) ++, the submatrix
    for the negative cell is --, the submatrix using the row of ++ and the column of -- is +-,
    and the submatrix using the row of -- and the column of ++ is -+. The values are
    ++ = -dhat + dtilde,
    -- = dhat + dtilde,
    +- = -dhat - dtilde,
    -+ = dhat - dtilde.

    In COO, where we have three 1D vectors (rowIndices, colIndices, values), we set a displacement
    that corresponds to the position of the leakage values in each "row" of the vector (every maxNNZInRow
    entries in the vector corresponds to a row in the matrix). There are four entries per surface
    (++, --, +-, -+), so entriesPerSurf = 4.

    In a cartesian mesh, each cell has up to three positive surfaces (that is, surfaces where the
    cell is positive) (north, up, right) and up to three negative surfaces (south, down, left).
    We store the maximum number of positive surfaces per cell in MAX_POS_SURF_PER_CELL (= 3).

    Therefore, the number of entries in a "row" the number of groups (scattering) + (entriesPerSurf * MAX_POS_SURF_PER_CELL).
    Note, for the values in the -+ and -- submatrices, the values aren't in the row of the ++ submatrix, but we
    store them in the the "row" (portion of the 1D vector) of the positive cell. The matrix assembler will take
    care of repeated index values and add the respective values together.

    Therefore, each 1D vector has the following layout:

                                    |-------- Scatter --------|  |----Leakage 0----| |----Leakage 1----| |----Leakage 2----|
    "Row"0 (Cell 0, ScatterFrom 0): [ScatterTo0, ... ScatterToN, ++0, -+0, --0, +-0, ++1, -+1, --1, +-1, ++2, -+2, --2, +-2,
    "Row"1 (Cell 0, ScatterFrom 1): [ScatterTo0, ... ScatterToN, ++0, -+0, --0, +-0, ++1, -+1, --1, +-1, ++2, -+2, --2, +-2,

    Where the leakage 0, 1, and 2 are the leakage through the first, second, and third surfaces in which the row cell is positive.
    // TODO/NOTE to future self: Perhaps a better pattern would be to store ALL scatterterms contiguously and then ALL leakage terms.
    // This may lead to a better data access pattern.

    Lastly, we need to consider leakage out of the system. where the positive cell is the exterior cell (-1).
    The above method did not account for this, and this leakage out of the system is only stored in the -- submatrix.
    Therefore, we add an additional nGroup entries for each positive leakage surface to the end of our 1D values vector.
    (That is, in total we add nGroups * nPosLeakageSurfs entries to the end of each COO vector.)
    */

   static constexpr size_t posPosDisplacement = 0;
   static constexpr size_t negPosDisplacement = 1;
   static constexpr size_t negNegDisplacement = 2;
   static constexpr size_t posNegDisplacement = 3;
   static constexpr PetscInt entriesPerSurf = 4;
   PetscInt maxNNZInRow = cmfdData.nGroups + entriesPerSurf * MAX_POS_SURF_PER_CELL;
   PetscInt matSize = cmfdData.nCells * cmfdData.nGroups;

   PetscFunctionBeginUser;

    // // There are a lot of options here for splitting up the matrix into submatrices per mpi rank
    // //  (on the PETSc side), i.e., # of rows/cols per rank and number of zeros on and off the diagonal (per row).
    PetscCall(MatCreateAIJKokkos(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
        matSize, matSize,
        PETSC_DEFAULT, NULL,
        PETSC_DEFAULT, NULL,
        &MMat));

    // Attempt. Could optimize with d_nnz based on scatter matrix layout (4th from last param)
    // MatCreateAIJKokkos(PETSC_COMM_WORLD, cmfdData.nGroups, cmfdData.nGroups,
    //     matSize, matSize, cmfdData.nGroups, NULL, 1, NULL, &MMat);

    static constexpr int method = 2;
    // METHOD 1: Use Vectors with emplace back to aPetscErrorCode storing zeros
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
                    if (isNonZero(value))
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
                if (isNonZero(value))
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
                    if (isNonZero(value)) {
                        rowIndices.emplace_back(posCellMatIdx);
                        colIndices.emplace_back(posCellMatIdx);
                        values.emplace_back(value);
                    }
                }
                if (negCellMatIdx >= 0)
                {
                    const PetscScalar value = dhat + dtilde;
                    if (isNonZero(value)) {
                        rowIndices.emplace_back(negCellMatIdx);
                        colIndices.emplace_back(negCellMatIdx);
                        values.emplace_back(value);
                    }
                }
                if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
                {
                    const PetscScalar value1 = -1 * dhat - dtilde;
                    if (isNonZero(value1)) {
                        rowIndices.emplace_back(posCellMatIdx);
                        colIndices.emplace_back(negCellMatIdx);
                        values.emplace_back(value1);
                    }

                    const PetscScalar value2 = dhat - dtilde;
                    if (isNonZero(value2)) {
                        rowIndices.emplace_back(negCellMatIdx);
                        colIndices.emplace_back(posCellMatIdx);
                        values.emplace_back(value2);
                    }
                }
            }
        }
        size_t numNonZero = values.size();


        PetscCall(MatSetPreallocationCOO(MMat, numNonZero, rowIndices.data(), colIndices.data()));
        PetscCall(MatSetValuesCOO(MMat, values.data(), ADD_VALUES));
    }
    else if constexpr(method == 2) // METHOD 2:
    // Use Kokkos views somewhat naively (use teams, scratch pad, etc. to optimize)
    // Just making sure this works for now
    // See the comment at the top of the method for a description of the layout
    {
        // Don't want to access self->cmfdData in the lambda, so copy it to a reference
        auto& _cmfdData = cmfdData;

        // (#26) Assume each row has maxNNZInRow non-zero entries
        PetscInt maxNNZEntries = maxNNZInRow * matSize + cmfdData.nPosLeakageSurfs * cmfdData.nGroups;

        // Maybe these get assembled on the GPU, copied to the host for MatSetValuesCOO
        // Maybe this changes based on where the matrix is assembled, or does Kokkos handle that?
        Kokkos::View<PetscInt *, AssemblyMemorySpace> rowIndices("rowIndicesCOO", maxNNZEntries);
        Kokkos::View<PetscInt *, AssemblyMemorySpace> colIndices("colIndicesCOO", maxNNZEntries);
        Kokkos::View<PetscScalar *, AssemblyMemorySpace> values("valuesCOO", maxNNZEntries);

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
            Kokkos::parallel_for("COOMatrixAssembler2: InLeakage", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<3>>({0, 0, 0}, {cmfdData.nGroups, cmfdData.nCells, MAX_POS_SURF_PER_CELL}),
                KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt posCellIdx, const PetscInt posSurfPosition)
            {
                // posSurfPosition is 0, 1, or 2 as we expect each cell to have three positive surfaces (MAX_POS_SURF_PER_CELL)

                // We are only on the diagonal of each block matrix,
                // so groupIdx could be from or to, but we use it to find the "row" (scatter from).

                const PetscInt posSurfIdx = _cmfdData.cell2PosSurf(posCellIdx, posSurfPosition);
                if (posSurfIdx >= 0) // -1 is invalid surface
                {
                    const PetscScalar dhat = _cmfdData.dHat(groupIdx, posSurfIdx);
                    const PetscScalar dtilde = _cmfdData.dTilde(groupIdx, posSurfIdx);

                    const PetscInt posCellMatIdx = (posCellIdx) * _cmfdData.nGroups + groupIdx;
                    const size_t locationStartIn1D = posCellMatIdx * maxNNZInRow + _cmfdData.nGroups + posSurfPosition * entriesPerSurf;

                    const PetscInt negCellIdx = _cmfdData.surf2Cell(posSurfIdx, 1);
                    const PetscInt negCellMatIdx = (negCellIdx) * _cmfdData.nGroups + groupIdx;

                    rowIndices(locationStartIn1D + posPosDisplacement) = posCellMatIdx;
                    colIndices(locationStartIn1D + posPosDisplacement) = posCellMatIdx;
                    values(locationStartIn1D + posPosDisplacement) = -1 * dhat + dtilde;

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

            // TODO: This is also not tightly nested and could be optimized
            Kokkos::parallel_for("COOMatrixAssembler2: OutLeakage", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<2>>({0, 0}, {cmfdData.nGroups, cmfdData.nPosLeakageSurfs}),
                KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt nthPosLeakageSurf)
            {
                const PetscInt posSurfIdx = _cmfdData.posLeakageSurfs(nthPosLeakageSurf);
                const PetscInt negCellIdx = _cmfdData.surf2Cell(posSurfIdx, 1);

                if (negCellIdx >= 0)
                {
                    const size_t locationIn1D = maxNNZInRow * matSize + nthPosLeakageSurf * _cmfdData.nGroups + groupIdx;
                    const PetscInt negCellMatIdx = (negCellIdx)*_cmfdData.nGroups + groupIdx;
                    const PetscScalar dhat = _cmfdData.dHat(groupIdx, posSurfIdx);
                    const PetscScalar dtilde = _cmfdData.dTilde(groupIdx, posSurfIdx);

                    rowIndices(locationIn1D) = negCellMatIdx;
                    colIndices(locationIn1D) = negCellMatIdx;
                    values(locationIn1D) = dhat + dtilde;
                }
            });
    }

        Kokkos::fence();
        PetscCall(MatSetPreallocationCOO(MMat, maxNNZEntries, rowIndices.data(), colIndices.data()));
        PetscCall(MatSetValuesCOO(MMat, values.data(), ADD_VALUES));
    }

    return PETSC_SUCCESS;
}

PetscErrorCode COOMatrixAssembler::_assembleFission(const FluxView& flux)
{
    auto& _cmfdData = cmfdData;

    // TODO (#26): Assuming no zeros in the vector. We can optimize based on chi to get the sparsity pattern.
    const PetscInt nnz = nRows;
    Kokkos::View<PetscInt *, AssemblyMemorySpace> rowIndices("rowIndicesKokkos", nnz);
    Kokkos::View<PetscScalar *, AssemblyMemorySpace> values("VecValuesKokkos", nnz);

    {
        // We create the functor beforehand to calclate the maximum team size
        auto functorVectorAssemble = KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<AssemblySpace>::member_type& teamMember)
        {
            const PetscInt cellIdx = teamMember.league_rank();

            PetscScalar cellFissionRate = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, _cmfdData.nGroups), [=] (const PetscInt fromGroupIdx, PetscScalar &localFissionRate)
            {
                // if FluxView is 2D
                // localFissionRate += cmfdData.nuFissionXS(fromGroupIdx, cellIdx) * flux(fromGroupIdx, cellIdx);

                // if FluxView is 1D. I think the data access pattern is bad
                localFissionRate += _cmfdData.nuFissionXS(fromGroupIdx, cellIdx) * flux(cellIdx * _cmfdData.nGroups + fromGroupIdx);
            }, cellFissionRate);

            teamMember.team_barrier();

            const PetscScalar localFissionRateVolume = cellFissionRate * _cmfdData.volume(cellIdx);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, _cmfdData.nGroups), [=] (const PetscInt toGroupIdx)
            {
                const PetscInt vecIdx = cellIdx * _cmfdData.nGroups + toGroupIdx;
                rowIndices(vecIdx) = vecIdx; // This is silly right now as we assume no zeros.
                values(vecIdx) = _cmfdData.chi(toGroupIdx, cellIdx) * localFissionRateVolume;
            });
        };

        // Use the team policy with the automatic team size to calculate the maximum team size
        Kokkos::TeamPolicy<AssemblySpace> nCellsRange(_cmfdData.nCells, Kokkos::AUTO);
        int maxTeamSize = nCellsRange.team_size_max(functorVectorAssemble, Kokkos::ParallelForTag());

        // If we can set the team size to the number of groups, we do so
        if (maxTeamSize >= _cmfdData.nGroups)
        {
            nCellsRange = Kokkos::TeamPolicy<AssemblySpace>(_cmfdData.nCells, _cmfdData.nGroups);
        }

        // Actually execute the parallel for loop
        Kokkos::parallel_for("COOVector", nCellsRange, functorVectorAssemble);
    }

    PetscFunctionBeginUser;
    Kokkos::fence();
    PetscCall(VecSetPreallocationCOO(fissionVec, nnz, rowIndices.data()));
    PetscCall(VecSetValuesCOO(fissionVec, values.data(), INSERT_VALUES));

    return PETSC_SUCCESS;
}

PetscErrorCode CSRMatrixAssembler::_assembleM()
{
    // Don't want to access self->cmfdData in the lambda, so copy it to a reference
    auto& _cmfdData = cmfdData;

    // See the comment in COOMatrixAssembler::_assembleM() for a similar description of the layout.
    // In COO, we store i, j, and values in three 1D vectors of length nnz.
    // In CSR, we store the column index and values in two 1D vectors of length nnz
    // and the index in the values vector where each row starts in 1D vector of length nRows + 1
    // (the last entry is the total number of non-zero entries in the matrix).
    // The biggest difference is that in CSR, we don't have the luxury of using INSERT_VALUES,
    // (which is used in COO to handle multiple entries to the same index), so we have to
    // deal with race conditions ourselves.
    //
    // Extra info: In COO, the three vectors didn't have to be in a specific order, but we made
    // them almost ordered by row (in reality, by cell, so -- are in the wrong "row" in the 1D vector).
    // In CSR, the order of column indices in a row isn't important, but all values and columns must be
    // grouped by row.

    // We use a slightly different layout. See COO's _assembleM() for the notation.
    //                                  |-------- Scatter --------| | Leakage +- | | Leakage -+ |
    // "Row"0 (Cell 0, ScatterFrom 0): [ScatterTo0, ... ScatterToN, +-0, +-1, +-2, -+0, -+1, -+2,
    // "Row"1 (Cell 0, ScatterFrom 1): [ScatterTo0, ... ScatterToN, +-0, +-1, +-2, -+0, -+1, -+2,
    //
    // Some notes:
    // - Rather than organizing the values by surface, we organize them by +- and -+.
    // - ++ and -- leakage (including out leakage) need to be added to the diagonals of the scatter
    //      submatrices (in our 1D vector) before we hand off to PETSc.
    // - The -+ leakage terms aren't stored on the iteration of "row 0" (the positive cell),
    //      but rather in the iteration in which "row0" is the negative cell.

    static constexpr size_t additionalEntriesPerSurf = 2;

    // On each row we have
    const PetscInt maxNNZInRow = cmfdData.nGroups + MAX_POS_SURF_PER_CELL * additionalEntriesPerSurf;
    const PetscInt matSize = cmfdData.nCells * cmfdData.nGroups; // row or col
    const PetscInt numNonZero = matSize * maxNNZInRow;

    // Specifying the memory space to AssemblyMemorySpace upsets PETSc,
    // even though AssemblyMemorySpace should end up being the same as the
    // default memory space. This is almost certainly a soft bug in PETSc.
    Kokkos::View<PetscInt *> rowIndices("rowIndicesKokkos", matSize + 1);
    Kokkos::View<PetscInt *> colIndices("colIndicesKokkos", numNonZero);
    Kokkos::View<PetscScalar *> values("valuesCOO", numNonZero);

    // Scatter XS and Removal XS won't conflict
    // Leakage terms will conflict on the ++ and -- terms.
    // There are a few options to handle this:
    // - Do atomics on ++ and -- terms
    // - Do ++, +-, and -+ terms and store --. Fence and then do -- terms atomically
    // - ... I'm sure there are more options.
    static constexpr int method = 1;

    { // Kokkos scope

        // TODO (#26): We specify the number of non-zero entries per row as the same for all rows.
        // This makes figuring out the row indices vector trivial
        Kokkos::parallel_for("CSRMatrixAssembler: RowIndices", Kokkos::RangePolicy<AssemblySpace>(0, matSize + 1),
            KOKKOS_LAMBDA(const PetscInt rowIdx)
        {
            // The first entry is always 0, the last entry is the total number of non-zero entries
            if (rowIdx == 0) { rowIndices(rowIdx) = 0; }
            else if (rowIdx == matSize) { rowIndices(rowIdx) = numNonZero; }
            else { rowIndices(rowIdx) = rowIdx * maxNNZInRow; }
        });

        // This is basically verbatim from the COOMatrixAssembler without the row index
        Kokkos::parallel_for("CSRMatrixAssembler: ScatterXS", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<3>>({0, 0, 0}, {_cmfdData.nGroups, _cmfdData.nGroups, _cmfdData.nCells}),
            KOKKOS_LAMBDA(const PetscInt scatterToIdx, const PetscInt scatterFromIdx, const PetscInt cellIdx)
        {
            // Don't need diagonal entries here because we use removal xs
            // Else, we would have to have another entry for the diagonal
            if (scatterToIdx != scatterFromIdx)
            {
                const size_t rowIdxInMat = cellIdx * _cmfdData.nGroups + scatterFromIdx;
                const size_t locationIn1D = rowIdxInMat * maxNNZInRow + scatterToIdx;

                colIndices(locationIn1D) = cellIdx * _cmfdData.nGroups + scatterToIdx;
                values(locationIn1D) = -1 * _cmfdData.scatteringXS(scatterToIdx, scatterFromIdx, cellIdx) * _cmfdData.volume(cellIdx);
            }
        });

        // This is also verbatim from the COOMatrixAssembler without the row index
        Kokkos::parallel_for("CSRMatrixAssembler: RemovalXS", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<2>>({0, 0}, {_cmfdData.nGroups, _cmfdData.nCells}),
            KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt cellIdx)
        {
            // groupIdx is both from and to since removal includes self scattering
            const size_t rowIdxInMat = cellIdx * _cmfdData.nGroups + groupIdx;
            const size_t locationIn1D = rowIdxInMat * maxNNZInRow + groupIdx;

            colIndices(locationIn1D) = rowIdxInMat; // diagonal
            values(locationIn1D) = _cmfdData.removalXS(groupIdx, cellIdx) * _cmfdData.volume(cellIdx);
        });

        // There are potential race conditions between the above kernel and the following kernels
        // as they both write to the diagonal of the matrix.
        Kokkos::fence(); // We could just make the values set above atomic. I'm not sure which is better.

        // METHOD 1: Use atomics on ++ and -- terms as they can refer to the same cell
        // on different iterations of the loop.
        if constexpr(method == 1)
        {
            const PetscInt nGroups = _cmfdData.nGroups;
            Kokkos::parallel_for("CSRMatrixAssembler1: InLeakage", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<3>>({0, 0, 0}, {_cmfdData.nGroups, _cmfdData.nCells, MAX_POS_SURF_PER_CELL}),
                KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt posCellIdx, const PetscInt posSurfPosition)
            {
                // posSurfPosition is 0, 1, or 2 as we expect each cell to have three positive surfaces (MAX_POS_SURF_PER_CELL)

                // We are only on the diagonal of each block matrix,
                // so groupIdx could be from or to, but we use it to find the "row" (scatter from).

                // TODO (kind of in line with #26): Cells on the boundary don't have less than three surfaces with a non-boundary cell,
                // but we currently store zeros in the +- and -+ values. These zeros could be removed.

                const PetscInt posSurfIdx = _cmfdData.cell2PosSurf(posCellIdx, posSurfPosition);
                if (posSurfIdx >= 0) // -1 is invalid surface
                {
                    const PetscScalar dhat = _cmfdData.dHat(groupIdx, posSurfIdx);
                    const PetscScalar dtilde = _cmfdData.dTilde(groupIdx, posSurfIdx);

                    const PetscInt posCellMatIdx = posCellIdx * _cmfdData.nGroups + groupIdx;
                    const PetscInt posRowBeginIn1D = posCellMatIdx * maxNNZInRow;

                    // ++ (in pos cell row, pos cell column)
                    const PetscInt posDiagIn1D = posRowBeginIn1D + groupIdx;
                    Kokkos::atomic_add(&values(posDiagIn1D), -1 * dhat + dtilde);

                    const PetscInt negCellIdx = _cmfdData.surf2Cell(posSurfIdx, 1);
                    if (negCellIdx >= 0)
                    {
                        const PetscInt negCellMatIdx = negCellIdx * _cmfdData.nGroups + groupIdx;
                        const PetscInt negRowBeginIn1D = negCellMatIdx * maxNNZInRow;

                        // +- (in pos cell row, neg cell column)
                        const PetscInt posNegIn1D = posRowBeginIn1D + _cmfdData.nGroups + posSurfPosition;
                        colIndices(posNegIn1D) = negCellMatIdx;
                        values(posNegIn1D) = -1 * dhat - dtilde;

                        // Get the "position" (0, 1, or 2) of the surface for the negative cell
                        // TODO: There are several ways to find the correct spot for -+ (precalculate the negative surface positions,
                        // maybe move -+ into a different kernel), but this works for now and can be optimized later.
                        // Note, if this is refactored, _cmfdData.cell2NegSurf can probably be removed.
                        PetscInt negSurfPosition = -1;
                        for (PetscInt testNegSurfPosition = 0; testNegSurfPosition < MAX_POS_SURF_PER_CELL; ++testNegSurfPosition)
                        {
                            if (_cmfdData.cell2NegSurf(negCellIdx, testNegSurfPosition) == posSurfIdx)
                            {
                                negSurfPosition = testNegSurfPosition;
                                break;
                            }
                        }

                        // -+ (in neg cell row, pos cell column)
                        const PetscInt negPosIn1D = negRowBeginIn1D + nGroups + MAX_POS_SURF_PER_CELL + negSurfPosition;
                        colIndices(negPosIn1D) = posCellMatIdx;
                        values(negPosIn1D) = dhat - dtilde;

                        // -- (in neg cell row, neg cell column)
                        const size_t negDiagIn1D = negRowBeginIn1D + groupIdx;
                        Kokkos::atomic_add(&values(negDiagIn1D), dhat + dtilde);
                    }
                }
            });
        }

        Kokkos::parallel_for("CSRMatrixAssembler1: OutLeakage", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<2>>({0, 0}, {cmfdData.nGroups, cmfdData.nPosLeakageSurfs}),
            KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt nthPosLeakageSurf)
        {
            const PetscInt posSurfIdx = _cmfdData.posLeakageSurfs(nthPosLeakageSurf);
            const PetscInt negCellIdx = _cmfdData.surf2Cell(posSurfIdx, 1);

            if (negCellIdx >= 0)
            {
                const PetscInt negCellMatIdx = (negCellIdx)*_cmfdData.nGroups + groupIdx;
                const PetscScalar dhat = _cmfdData.dHat(groupIdx, posSurfIdx);
                const PetscScalar dtilde = _cmfdData.dTilde(groupIdx, posSurfIdx);

                const PetscInt negDiagIn1D = (negCellMatIdx * maxNNZInRow) + groupIdx;

                // The posLeakage surfaces may not be unique so we use atomics
                Kokkos::atomic_add(&values(negDiagIn1D), dhat + dtilde);
            }
        });

    } // Kokkos scope ends

    PetscFunctionBeginUser;
    // TODO (#26): The null allows you to specify the number of non-zero entries per row.
    // This will improve memory/performance if implemented
    PetscCall(MatCreateSeqAIJKokkos(PETSC_COMM_WORLD, matSize, matSize, numNonZero, NULL, &MMat));

    // Actually put the values into the matrix
    Kokkos::fence();
    PetscCall(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, matSize, matSize, rowIndices, colIndices, values, &MMat));

    return PETSC_SUCCESS;
}

PetscErrorCode CSRMatrixAssembler::_assembleFission(const FluxView& flux)
{
    auto& _cmfdData = cmfdData;
    auto& _values = _fissionVectorView;

    // The following is nearly identical to the COOMatrixAssembler::_assembleFission method,
    // but I anticipate that the COOMatrixAssembler version will evolve differently (it can handle sparsity,
    // while the CSRMatrixAssembler version will not).
    {
        // We create the functor beforehand to calclate the maximum team size
        auto functorVectorAssemble = KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<AssemblySpace>::member_type& teamMember)
        {
            const PetscInt cellIdx = teamMember.league_rank();

            PetscScalar cellFissionRate = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, _cmfdData.nGroups), [=] (const PetscInt fromGroupIdx, PetscScalar &localFissionRate)
            {
                // if FluxView is 2D
                // localFissionRate += cmfdData.nuFissionXS(fromGroupIdx, cellIdx) * flux(fromGroupIdx, cellIdx);

                // if FluxView is 1D. I think the data access pattern is bad
                localFissionRate += _cmfdData.nuFissionXS(fromGroupIdx, cellIdx) * flux(cellIdx * _cmfdData.nGroups + fromGroupIdx);
            }, cellFissionRate);

            teamMember.team_barrier();

            const PetscScalar localFissionRateVolume = cellFissionRate * _cmfdData.volume(cellIdx);

            Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, _cmfdData.nGroups), [=] (const PetscInt toGroupIdx)
            {
                const PetscInt vecIdx = cellIdx * _cmfdData.nGroups + toGroupIdx;
                _values(vecIdx) = _cmfdData.chi(toGroupIdx, cellIdx) * localFissionRateVolume;
            });
        };

        // Use the team policy with the automatic team size to calculate the maximum team size
        Kokkos::TeamPolicy<AssemblySpace> nCellsRange(_cmfdData.nCells, Kokkos::AUTO);
        int maxTeamSize = nCellsRange.team_size_max(functorVectorAssemble, Kokkos::ParallelForTag());

        // If we can set the team size to the number of groups, we do so
        if (maxTeamSize >= _cmfdData.nGroups)
        {
            nCellsRange = Kokkos::TeamPolicy<AssemblySpace>(_cmfdData.nCells, _cmfdData.nGroups);
        }

        // Actually execute the parallel for loop
        Kokkos::parallel_for("CSRVector", nCellsRange, functorVectorAssemble);
    }

    PetscFunctionBeginUser;
    Kokkos::fence();



    // The second input parameter is the block size, which I believe should be 1.
    PetscCall(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF, 1, nRows, _fissionVectorView.data(), &fissionVec));
    // Maybe VecCreateMPIKokkosWithArray is better?
    return PETSC_SUCCESS;
}