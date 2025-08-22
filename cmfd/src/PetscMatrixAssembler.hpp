#pragma once

#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"
#include "Kokkos_Core.hpp"
#include "highfive/H5File.hpp"
#include <variant>

#include "CMFDData.hpp"

// Parent type for interfacing with the Mat and Vec without
// knowledge of the template type. Also hides methods that
// should be private.
struct MatrixAssemblerInterface
{
    PetscInt nRows = 0;

    // Returns the "M" matrix that includes leakage/removal/scattering
    virtual Mat getM() const = 0;

    // Returns the "f" vector that includes the fission source term.
    virtual Vec getFissionSource(const Vec& fluxPetsc) = 0;

    // Create a vector with the same type and size as fissionVec
    virtual PetscErrorCode instantiateVec(Vec &vec) const = 0;
};

// Parent class for all PETSc matrix assemblers.
// Templating allows avoiding allocating views on the host space if it is not needed.
template <typename AssemblySpace = Kokkos::DefaultExecutionSpace>
struct PetscMatrixAssembler : public MatrixAssemblerInterface
{
    using AssemblyMemorySpace = typename AssemblySpace::memory_space;
    static_assert(Kokkos::is_memory_space<AssemblyMemorySpace>::value,
                  "AssemblyMemorySpace must be a Kokkos memory space");

    using CMFDDataType = CMFDData<AssemblyMemorySpace>;
    using FluxView = typename CMFDDataType::View1D;

    CMFDDataType cmfdData;
    FluxView flux; // Kokkos view for the flux Vec

    // Returned data
    Vec fissionVec;
    Mat MMat;

    PetscMatrixAssembler() = default;
    PetscMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : cmfdData(CMFDCoarseMesh)
    {
        // Create the vector (not allocated yet)
        PetscCallCXXAbort(PETSC_COMM_SELF, VecCreate(PETSC_COMM_WORLD, &fissionVec));

        // Set the type of the vector (etc.) based on PETSc CLI options.
        // Default is AIJ sparse matrix.
        PetscCallCXXAbort(PETSC_COMM_SELF, VecSetFromOptions(fissionVec));

        // We are always using the Kokkos Vec type so data are
        // are accessible as views.
        PetscCallCXXAbort(PETSC_COMM_SELF, VecSetType(fissionVec, VECKOKKOS));

        // Set the vector dimensions (just for compatibility checks))
        // The PETSC_DECIDEs are for sub matrices split across multiple MPI ranks.
        nRows = cmfdData.nCells * cmfdData.nGroups;
        PetscCallCXXAbort(PETSC_COMM_SELF, VecSetSizes(fissionVec, PETSC_DECIDE, nRows));

        // We defer Mat creation to _assembleM() which is called by the derived class constructor
    }

    ~PetscMatrixAssembler()
    {
        PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&fissionVec));
        PetscCallCXXAbort(PETSC_COMM_SELF, MatDestroy(&MMat));
    }

    Mat getM() const final override
    {
        return MMat;
    }

    Vec getFissionSource(const Vec& fluxPetsc) final override
    {
        VecGetKokkosView(fluxPetsc, &flux);
        _assembleFission(flux);
        VecRestoreKokkosView(fluxPetsc, &flux);
        return fissionVec;
    }

    PetscErrorCode instantiateVec(Vec &vec) const final override
    {
        PetscCall(VecDuplicate(fissionVec, &vec));
        return PETSC_SUCCESS;
    }

    // These should be private, but that can't be because you
    // can't have host lambda functions in a private method.
    // These are protected when a pointer to the interface base
    // class is used.
    virtual PetscErrorCode _assembleM() = 0;
    virtual PetscErrorCode _assembleFission(const FluxView& flux) = 0;
};

// Uses Mat/VecSetValue(s) to naively assemble a matrix in PETSc. The focus is on accuracy over performance.
struct SimpleMatrixAssembler : public PetscMatrixAssembler<Kokkos::DefaultHostExecutionSpace>
{
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using AssemblyMemorySpace = Kokkos::HostSpace;

    SimpleMatrixAssembler() = default;
    SimpleMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());
        }

    PetscErrorCode _assembleM() override;
    PetscErrorCode _assembleFission(const FluxView& flux) override;
};

// Uses Mat/VecSetValueCOO to assemble a matrix in PETSc.
struct COOMatrixAssembler : public PetscMatrixAssembler<Kokkos::DefaultHostExecutionSpace>
{
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using AssemblyMemorySpace = Kokkos::HostSpace;

    COOMatrixAssembler() = default;
    COOMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());
        }

    PetscErrorCode _assembleM() override;
    PetscErrorCode _assembleFission(const FluxView& flux) override;
};

// Uses Kokkos views to assemble a matrix in PETSc (CSR Format)
struct CSRMatrixAssembler : public PetscMatrixAssembler<>
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;
    using AssemblyMemorySpace = AssemblySpace::memory_space;

    CSRMatrixAssembler() = default;
    CSRMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());

            _fissionVectorView = Kokkos::View<PetscScalar *, AssemblyMemorySpace>("VecValuesKokkos", nRows);
        }

    PetscErrorCode _assembleM() override;
    PetscErrorCode _assembleFission(const FluxView& flux) override;

    // The class needs to own the fission vector view because it should have
    // the same lifetime/scope as the corresponding PetscVec. The PetscVec
    // does not control the lifetime, and if the view is destroyed, errors
    // will occur when the PetscVec is accessed.
    Kokkos::View<PetscScalar *, AssemblyMemorySpace> _fissionVectorView;

};
