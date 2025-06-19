#pragma once

#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"
#include "Kokkos_Core.hpp"
#include "highfive/H5File.hpp"

#include "CMFDData.hpp"

// Parent class for all PETSc matrix assemblers.
// Templating allows avoiding allocating views on the host space if it is not needed.
template <typename AssemblySpace = Kokkos::DefaultExecutionSpace>
struct PetscMatrixAssembler
{
    using AssemblyMemorySpace = typename AssemblySpace::memory_space;
    static_assert(Kokkos::is_memory_space<AssemblyMemorySpace>::value,
                  "AssemblyMemorySpace must be a Kokkos memory space");

    using CMFDDataType = CMFDData<AssemblyMemorySpace>;
    using View2D = typename CMFDDataType::View2D;

    CMFDDataType cmfdData;

    PetscMatrixAssembler() = default;
    PetscMatrixAssembler(const HighFive::Group &CMFDCoarseMesh) : cmfdData(CMFDCoarseMesh) {};

    // Returns the "M" matrix that includes leakage/removal/scattering
    virtual Mat assembleM() const = 0;

    // Returns the "F" vector without flux
    virtual Vec assembleF(const View2D& flux) const = 0;
};

// Uses Mat/VecSetValue(s) to naively assemble a matrix in PETSc. The focus is on accuracy over performance.
struct SimpleMatrixAssembler : public PetscMatrixAssembler<Kokkos::DefaultHostExecutionSpace>
{
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using AssemblyMemorySpace = Kokkos::HostSpace;
    using PetscMatrixAssembler<AssemblySpace>::PetscMatrixAssembler; // Inherit constructors
    Mat assembleM() const override;
    Vec assembleF(const View2D& flux) const override;
};

// Uses Mat/VecSetValueCOO to assemble a matrix in PETSc.
struct COOMatrixAssembler : public PetscMatrixAssembler<Kokkos::DefaultHostExecutionSpace>
{
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using AssemblyMemorySpace = Kokkos::HostSpace;
    using PetscMatrixAssembler<AssemblySpace>::PetscMatrixAssembler;
    Mat assembleM() const override;
    Vec assembleF(const View2D& flux) const override;
};

// Uses Kokkos views to assemble a matrix in PETSc (CSR Format)
struct KokkosMatrixAssembler : public PetscMatrixAssembler<>
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;
    using AssemblyMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;
    using PetscMatrixAssembler<>::PetscMatrixAssembler;
    Mat assembleM() const override;
    Vec assembleF(const View2D& flux) const override;
};
