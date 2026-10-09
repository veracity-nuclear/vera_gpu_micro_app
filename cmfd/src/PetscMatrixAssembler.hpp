#pragma once

#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"
#include "Kokkos_Core.hpp"
#include "highfive/H5File.hpp"
#include <utility>
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

    using CMFDDataType = CMFDData<AssemblySpace>;
    using FluxView = typename CMFDDataType::View1D;

    CMFDDataType cmfdData;
    FluxView flux; // Kokkos view for the flux Vec

    // Returned data
    Vec fissionVec = nullptr;
    Mat MMat = nullptr;

    PetscErrorCode initialize()
    {
        PetscFunctionBegin;
        // Create the vector (not allocated yet)
        PetscCall(VecCreate(PETSC_COMM_WORLD, &fissionVec));

        // Set the type of the vector (etc.) based on PETSc CLI options.
        // Default is AIJ sparse matrix.
        PetscCall(VecSetFromOptions(fissionVec));

        // We are always using the Kokkos Vec type so data are
        // are accessible as views.
        PetscCall(VecSetType(fissionVec, VECKOKKOS));

        // Set the vector dimensions (just for compatibility checks))
        // The PETSC_DECIDEs are for sub matrices split across multiple MPI ranks.
        nRows = cmfdData.nCells * cmfdData.nGroups;
        PetscCall(VecSetSizes(fissionVec, PETSC_DECIDE, nRows));

        PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscMatrixAssembler() = default;
    PetscMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : cmfdData(CMFDCoarseMesh)
    {
        PetscCallCXXAbort(PETSC_COMM_SELF, initialize());
    }
    PetscMatrixAssembler(CMFDDataType &&data)
        : cmfdData(std::move(data))
    {
        PetscCallCXXAbort(PETSC_COMM_SELF, initialize());
    }

    // Vec/Mat are reference-counted PETSc handles that this class destroys. A member-wise
    // copy would destroy them twice (and leave the copy dangling), so copying is disabled
    // and moving transfers ownership (VecDestroy/MatDestroy on a null handle are no-ops).
    PetscMatrixAssembler(const PetscMatrixAssembler &) = delete;
    PetscMatrixAssembler &operator=(const PetscMatrixAssembler &) = delete;
    PetscMatrixAssembler(PetscMatrixAssembler &&other) noexcept
        : MatrixAssemblerInterface(other),
          cmfdData(std::move(other.cmfdData)),
          flux(std::move(other.flux)),
          fissionVec(std::exchange(other.fissionVec, nullptr)),
          MMat(std::exchange(other.MMat, nullptr))
    {
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
    using CMFDDataType = CMFDData<AssemblySpace>;

    SimpleMatrixAssembler() = default;
    SimpleMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());
        }
    SimpleMatrixAssembler(CMFDDataType &&data)
        : PetscMatrixAssembler<AssemblySpace>(std::move(data)) {
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
    using CMFDDataType = CMFDData<AssemblySpace>;

    COOMatrixAssembler() = default;
    COOMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());
            PetscCallCXXAbort(PETSC_COMM_SELF, _preallocateFission());
        }
    COOMatrixAssembler(CMFDDataType &&data)
        : PetscMatrixAssembler<AssemblySpace>(std::move(data)) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());
            PetscCallCXXAbort(PETSC_COMM_SELF, _preallocateFission());
        }

    PetscErrorCode _assembleM() override;
    PetscErrorCode _assembleFission(const FluxView& flux) override;

    // The COO sparsity pattern of the fission vector does not change between power iterations,
    // so VecSetPreallocationCOO (which builds PETSc's COO maps) is done once here and
    // _assembleFission only calls VecSetValuesCOO with the reused _fissionValues buffer.
    PetscErrorCode _preallocateFission();
    Kokkos::View<PetscScalar *, AssemblyMemorySpace> _fissionValues;
};

// Uses Kokkos views to assemble a matrix in PETSc (CSR Format)
struct CSRMatrixAssembler : public PetscMatrixAssembler<>
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;
    using AssemblyMemorySpace = AssemblySpace::memory_space;
    using CMFDDataType = CMFDData<AssemblySpace>;

    CSRMatrixAssembler() = default;
    CSRMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());
            PetscCallCXXAbort(PETSC_COMM_SELF, _createFissionVec());
        }
    CSRMatrixAssembler(CMFDDataType &&data)
        : PetscMatrixAssembler<AssemblySpace>(std::move(data)) {
            PetscCallCXXAbort(PETSC_COMM_SELF, _assembleM());
            PetscCallCXXAbort(PETSC_COMM_SELF, _createFissionVec());
        }

    PetscErrorCode _assembleM() override;
    PetscErrorCode _assembleFission(const FluxView& flux) override;

    // Replaces the generic Vec from initialize() with a sequential Kokkos Vec (matching the
    // sequential MMat) that wraps _fissionVectorView. It is created once and refilled in place
    // by _assembleFission, so callers may cache the Vec returned by getFissionSource().
    PetscErrorCode _createFissionVec();

    // The class needs to own the fission vector view because it should have
    // the same lifetime/scope as the corresponding PetscVec. The PetscVec
    // does not control the lifetime, and if the view is destroyed, errors
    // will occur when the PetscVec is accessed.
    Kokkos::View<PetscScalar *, AssemblyMemorySpace> _fissionVectorView;

};
