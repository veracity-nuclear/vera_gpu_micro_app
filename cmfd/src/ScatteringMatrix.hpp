// Construct the scattering matrix (either dense or COO/CSR sparse) from the format MPACT uses
#pragma once

#include <petscvec_kokkos.hpp>
#include <Kokkos_Core.hpp>

template <typename AssemblySpace = Kokkos::DefaultExecutionSpace>
struct ScatteringMatrix
{
    using MemorySpace = typename AssemblySpace::memory_space;
    static_assert(Kokkos::is_memory_space<MemorySpace>::value,
                  "MemorySpace must be a Kokkos memory space");

    using View1D = Kokkos::View<PetscScalar *, MemorySpace>;
    using View2D = Kokkos::View<PetscScalar **, MemorySpace>;
    using View3D = Kokkos::View<PetscScalar ***, MemorySpace>;

    View1D cellNumberMap, scatterFromMap, scatterToMap;

    size_t nCells{}, nGroups{}, nValues{};

    ScatteringMatrix() = default;

    ScatteringMatrix(View2D gMin, View2D gMax, size_t nValues)
        : nValues(nValues)
    {
        nCells  = gMin.extent(0);
        nGroups = gMin.extent(1);

        cellNumberMap = View1D("cellNumberMap", nValues);
        scatterFromMap = View1D("scatterFromMap", nValues);
        scatterToMap = View1D("scatterToMap", nValues);

        auto h_cellNumberMap = Kokkos::create_mirror_view(cellNumberMap);
        auto h_scatterFromMap = Kokkos::create_mirror_view(scatterFromMap);
        auto h_scatterToMap = Kokkos::create_mirror_view(scatterToMap);

        size_t index1D = 0;
        for (size_t cellIdx = 0; cellIdx < nCells; cellIdx++)
        {
            for (size_t scatterTo = 0; scatterTo < nGroups; scatterTo++)
            {
                // gMin and gMax are 1-based indexing, so subtract 1
                for (size_t scatterFrom = gMin(cellIdx, scatterTo) - 1;
                     scatterFrom < gMax(cellIdx, scatterTo); scatterFrom++)
                {
                    h_cellNumberMap(index1D) = cellIdx;
                    h_scatterFromMap(index1D) = scatterFrom;
                    h_scatterToMap(index1D) = scatterTo;
                    index1D++;
                }
            }
        }

        Kokkos::deep_copy(cellNumberMap, h_cellNumberMap);
        Kokkos::deep_copy(scatterFromMap, h_scatterFromMap);
        Kokkos::deep_copy(scatterToMap, h_scatterToMap);
    }

    View3D constructDense(View1D scattering1D)
    {
        assert (scattering1D.extent(0) == nValues);

        auto _cellNumberMap = cellNumberMap;
        auto _scatterFromMap = scatterFromMap;
        auto _scatterToMap = scatterToMap;

        View3D scatteringXS("scatteringXS", nGroups, nGroups, nCells);
        Kokkos::deep_copy(scatteringXS, 0.0); // Initialize to zero

        Kokkos::parallel_for("constructDenseScatteringMatrix", Kokkos::RangePolicy<AssemblySpace>(0, nValues),
            KOKKOS_LAMBDA(const size_t i)
            {
                const size_t scatterFrom = _scatterFromMap(i);
                const size_t scatterTo = _scatterToMap(i);
                const size_t cellIdx = _cellNumberMap(i);
                scatteringXS(scatterFrom, scatterTo, cellIdx) = scattering1D(i);
            });

        Kokkos::fence();
        return scatteringXS;
    }

};