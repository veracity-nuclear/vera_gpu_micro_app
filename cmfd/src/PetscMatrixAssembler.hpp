#pragma once

#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"
#include "Kokkos_Core.hpp"
#include "highfive/H5File.hpp"

#include "CMFDData.hpp"

// Parent class for all PETSc matrix assemblers.
// Templating allows avoiding allocating views on the host space if it is not needed.
template <typename AssemblyMemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct PetscMatrixAssembler
{
    static_assert(Kokkos::is_memory_space<AssemblyMemorySpace>::value,
                  "AssemblyMemorySpace must be a Kokkos memory space");

    using CMFDDataType = CMFDData<AssemblyMemorySpace>;

    CMFDDataType cmfdData;

    PetscMatrixAssembler(const HighFive::Group &CMFDCoarseMesh) : cmfdData(CMFDCoarseMesh) {};

    virtual Mat assemble() const = 0;

};

// Uses Mat/VecSetValue(s) to assemble a matrix in PETSc.
struct SimpleMatrixAssembler : public PetscMatrixAssembler<Kokkos::HostSpace>
{
    SimpleMatrixAssembler(const HighFive::Group& CMFDCoarseMesh) : PetscMatrixAssembler(CMFDCoarseMesh) {}
    Mat assemble() const override;
};

// Uses Mat/VecSetValueCOO to assemble a matrix in PETSc.
struct COOMatrixAssembler : public PetscMatrixAssembler<Kokkos::HostSpace>
{
    COOMatrixAssembler(const HighFive::Group& CMFDCoarseMesh) : PetscMatrixAssembler(CMFDCoarseMesh) {}
    Mat assemble() const override;
};

// Uses Kokkos views to assemble a matrix in PETSc (CSR Format)
struct KokkosMatrixAssembler : public PetscMatrixAssembler<>
{
    KokkosMatrixAssembler(const HighFive::Group& CMFDCoarseMesh) : PetscMatrixAssembler(CMFDCoarseMesh) {}
    Mat assemble() const override;
};
