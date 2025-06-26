#include "PetscMatrixAssembler.hpp"

inline bool isNonZero(const PetscScalar& value)
{
    return std::abs(value) > PETSC_MACHINE_EPSILON;
}

void SimpleMatrixAssembler::_assembleM()
{
    PetscFunctionBeginUser;

    // Create the matrix (not allocated yet)
    MatCreate(PETSC_COMM_WORLD, &MMat);

    // Set the matrix dimensions (just for compatibility checks))
    // The PETSC_DECIDEs are for sub matrices split across multiple MPI ranks.
    const PetscInt matSize = cmfdData.nCells * cmfdData.nGroups;
    MatSetSizes(MMat, PETSC_DECIDE, PETSC_DECIDE, matSize, matSize);

    // Set the type of the matrix (etc.) based on PETSc CLI options.
    // Default is AIJ sparse matrix.
    MatSetFromOptions(MMat);

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
                    MatSetValue(MMat, cellIdx * cmfdData.nGroups + scatterFromIdx, cellIdx * cmfdData.nGroups + scatterToIdx, value, ADD_VALUES);
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
                MatSetValue(MMat, diagIdx, diagIdx, value, ADD_VALUES);
            }
            {
                const PetscInt diagIdx = cellIdx * cmfdData.nGroups + groupIdx;
                MatSetValue(MMat, diagIdx, diagIdx, value, ADD_VALUES);
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
                if (isNonZero(value)) {MatSetValue(MMat, posCellMatIdx, posCellMatIdx, value, ADD_VALUES);}
            }
            if (negCellMatIdx >= 0)
            {
                const PetscScalar value = dhat + dtilde;
                if (isNonZero(value)) {MatSetValue(MMat, negCellMatIdx, negCellMatIdx, value, ADD_VALUES);}
            }
            if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
            {
                const PetscScalar value1 = -1 * dhat - dtilde;
                if (isNonZero(value1)) {MatSetValue(MMat, posCellMatIdx, negCellMatIdx, value1, ADD_VALUES);}

                const PetscScalar value2 = dhat - dtilde;
                if (isNonZero(value2)) {MatSetValue(MMat, negCellMatIdx, posCellMatIdx, value2, ADD_VALUES);}
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
                    MatSetValue(MMat, cellIdx * cmfdData.nGroups + scatterFromIdx, cellIdx * cmfdData.nGroups + scatterToIdx, value, ADD_VALUES);
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
                MatSetValue(MMat, diagIdx, diagIdx, value, ADD_VALUES);
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
                if (isNonZero(value)) {MatSetValue(MMat, posCellMatIdx, posCellMatIdx, value, ADD_VALUES);}
            }
            if (negCellMatIdx >= 0)
            {
                const PetscScalar value = dhat + dtilde;
                if (isNonZero(value)) {MatSetValue(MMat, negCellMatIdx, negCellMatIdx, value, ADD_VALUES);}
            }
            if (posCellMatIdx >= 0 && negCellMatIdx >= 0)
            {
                const PetscScalar value1 = -1 * dhat - dtilde;
                if (isNonZero(value1)) {MatSetValue(MMat, posCellMatIdx, negCellMatIdx, value1, ADD_VALUES);}

                const PetscScalar value2 = dhat - dtilde;
                if (isNonZero(value2)) {MatSetValue(MMat, negCellMatIdx, posCellMatIdx, value2, ADD_VALUES);}
            }
        }
    }
    #endif

    // Actually put the values into the matrix
    MatAssemblyBegin(MMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(MMat, MAT_FINAL_ASSEMBLY);
}

void SimpleMatrixAssembler::_assembleFission(const FluxView& flux)
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
    VecSetValues(fissionVec, vecIndices.size(), vecIndices.data(), vecValues.data(), INSERT_VALUES);
    VecAssemblyBegin(fissionVec);
    VecAssemblyEnd(fissionVec);
}

void COOMatrixAssembler::_assembleM()
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
    MatCreateAIJKokkos(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
        matSize, matSize,
        PETSC_DEFAULT, NULL,
        PETSC_DEFAULT, NULL,
        &MMat);

    // Attempt. Could optimize with d_nnz based on scatter matrix layout (4th from last param)
    // MatCreateAIJKokkos(PETSC_COMM_WORLD, cmfdData.nGroups, cmfdData.nGroups,
    //     matSize, matSize, cmfdData.nGroups, NULL, 1, NULL, &MMat);

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


        MatSetPreallocationCOO(MMat, numNonZero, rowIndices.data(), colIndices.data());
        MatSetValuesCOO(MMat, values.data(), ADD_VALUES);
    }
    else if constexpr(method == 2) // METHOD 2:
    // Use Kokkos views somewhat naively (use teams, scratch pad, etc. to optimize)
    // Just making sure this works for now
    // See the comment at the top of the method for a description of the layout
    {
        // Don't want to access self->cmfdData in the lambda, so copy it to a reference
        auto& _cmfdData = cmfdData;

        // Assume each row has maxNNZInRow non-zero entries
        PetscInt maxNNZEntries = maxNNZInRow * matSize + cmfdData.nPosLeakageSurfs * cmfdData.nGroups;

        // Maybe these get assembled on the GPU, copied to the host for MatSetValuesCOO
        // Maybe this changes based on where the matrix is assembled, or does Kokkos handle that?
        Kokkos::View<PetscInt *, AssemblyMemorySpace> rowIndices("rowIndicesKokkos", maxNNZEntries);
        Kokkos::View<PetscInt *, AssemblyMemorySpace> colIndices("colIndicesKokkos", maxNNZEntries);
        Kokkos::View<PetscScalar *, AssemblyMemorySpace> values("valuesKokkos", maxNNZEntries);

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

            }
        );
    }

        MatSetPreallocationCOO(MMat, maxNNZEntries, rowIndices.data(), colIndices.data());
        MatSetValuesCOO(MMat, values.data(), ADD_VALUES);
    }
}

void COOMatrixAssembler::_assembleFission(const FluxView& flux)
{
    auto& _cmfdData = cmfdData;

    // TODO: Assuming no zeros in the vector. We can optimize based on chi to get the sparsity pattern.
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

        Kokkos::TeamPolicy<AssemblySpace> nCellsRange(_cmfdData.nCells, _cmfdData.nGroups);
        int maxTeamSize = nCellsRange.team_size_max(functorVectorAssemble, Kokkos::ParallelReduceTag());
        int teamSizeRecommended = nCellsRange.team_size_recommended(functorVectorAssemble, Kokkos::ParallelReduceTag());
        std::cout << "******************COOMatrixAssembler: maxTeamSize = " << maxTeamSize << ", teamSizeRecommended = " << teamSizeRecommended << std::endl;
        if (maxTeamSize > _cmfdData.nGroups)
        {
            // If the team size is too large, we need to reduce it
            nCellsRange = Kokkos::TeamPolicy<AssemblySpace>(_cmfdData.nCells, Kokkos::AUTO);
        }
        Kokkos::parallel_for("COOVector", nCellsRange, functorVectorAssemble);
    }

    PetscFunctionBeginUser;
    VecSetPreallocationCOO(fissionVec, nnz, rowIndices.data());
    VecSetValuesCOO(fissionVec, values.data(), INSERT_VALUES);
}