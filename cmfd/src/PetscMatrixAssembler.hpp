#pragma once

#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"
#include "Kokkos_Core.hpp"
#include "highfive/H5File.hpp"
#include <variant>

#include "CMFDData.hpp"

// Parent type for interfacing without the template type.
struct MatrixAssemblerInterface
{
    // Returns the "M" matrix that includes leakage/removal/scattering
    virtual Mat getM() const = 0;

    // Returns the "f" vector that includes the fission source term.
    virtual Vec getFissionSource(const Vec& fluxPetsc) = 0;
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
    FluxView flux;
    PetscInt nRows = 0;
    Vec fissionVec;
    Mat MMat;

    PetscMatrixAssembler() = default;
    PetscMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : cmfdData(CMFDCoarseMesh)
    {
        // Create the vector (not allocated yet)
        VecCreate(PETSC_COMM_WORLD, &fissionVec);

        // Set the type of the vector (etc.) based on PETSc CLI options.
        // Default is AIJ sparse matrix.
        VecSetFromOptions(fissionVec);

        // We are always using the Kokkos Vec type so data are
        // are accessible as views.
        VecSetType(fissionVec, VECKOKKOS);

        // Set the vector dimensions (just for compatibility checks))
        // The PETSC_DECIDEs are for sub matrices split across multiple MPI ranks.
        nRows = cmfdData.nCells * cmfdData.nGroups;
        VecSetSizes(fissionVec, PETSC_DECIDE, nRows);

        // We defer Mat creation to _assembleM() which is called by the derived class constructor
    }

    ~PetscMatrixAssembler()
    {
        VecDestroy(&fissionVec);
        MatDestroy(&MMat);
    }

    Mat getM() const final override
    {
        return MMat;
    }

    Vec getFissionSource(const Vec& fluxPetsc) final override
    {
        VecGetKokkosView(fluxPetsc, &flux);
        _assembleFission(flux);
        return fissionVec;
    }

    // These should be private, but that can't be because you
    // can't have host lambda functions in a private method.
    // These are protected when a pointer to the interface base
    // class is used.
    virtual void _assembleM() = 0;
    virtual void _assembleFission(const FluxView& flux) = 0;
};

// Uses Mat/VecSetValue(s) to naively assemble a matrix in PETSc. The focus is on accuracy over performance.
struct SimpleMatrixAssembler : public PetscMatrixAssembler<Kokkos::DefaultHostExecutionSpace>
{
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using AssemblyMemorySpace = Kokkos::HostSpace;

    SimpleMatrixAssembler() = default;
    SimpleMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            _assembleM();
        }

    void _assembleM() override;
    void _assembleFission(const FluxView& flux) override;
};

// Uses Mat/VecSetValueCOO to assemble a matrix in PETSc.
struct COOMatrixAssembler : public PetscMatrixAssembler<Kokkos::DefaultHostExecutionSpace>
{
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using AssemblyMemorySpace = Kokkos::HostSpace;

    COOMatrixAssembler() = default;
    COOMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            _assembleM();
        }

    void _assembleM() override;
    void _assembleFission(const FluxView& flux) override;
};

// Uses Kokkos views to assemble a matrix in PETSc (CSR Format)
struct KokkosMatrixAssembler : public PetscMatrixAssembler<>
{
    using AssemblySpace = Kokkos::DefaultExecutionSpace;
    using AssemblyMemorySpace = Kokkos::DefaultExecutionSpace::memory_space;

    KokkosMatrixAssembler() = default;
    KokkosMatrixAssembler(const HighFive::Group &CMFDCoarseMesh)
        : PetscMatrixAssembler<AssemblySpace>(CMFDCoarseMesh) {
            _assembleM();
        }

    void _assembleM() override;
    void _assembleFission(const FluxView& flux) override;
};
