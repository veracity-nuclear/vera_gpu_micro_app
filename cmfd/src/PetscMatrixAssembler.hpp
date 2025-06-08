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

    using DualView1D = Kokkos::DualView<PetscScalar *, AssemblyMemorySpace>;
    using DualView2D = Kokkos::DualView<PetscScalar **, AssemblyMemorySpace>;
    using DualViewSurf2Cell = Kokkos::DualView<PetscInt *[2], AssemblyMemorySpace>;
    using DualView3D = Kokkos::DualView<PetscScalar ***, AssemblyMemorySpace>;

    size_t nCells, nSurfaces, nGroups;

    DualView1D volume;
    DualView2D chi;
    DualView2D dHat;
    DualView2D dTilde;
    DualView2D nuFissionXS;
    DualView2D pastFlux;
    DualView2D removalXS;
    DualViewSurf2Cell surf2Cell;
    DualView3D scatteringXS;

    PetscMatrixAssembler(const HighFive::Group& CMFDCoarseMesh){
        volume = HDF5ToKokkosView<DualView1D>(CMFDCoarseMesh.getDataSet("volume"), "volume");
        chi = HDF5ToKokkosView<DualView2D>(CMFDCoarseMesh.getDataSet("chi"), "chi");
        dHat = HDF5ToKokkosView<DualView2D>(CMFDCoarseMesh.getDataSet("Dhat"), "Dhat");
        dTilde = HDF5ToKokkosView<DualView2D>(CMFDCoarseMesh.getDataSet("Dtilde"), "Dtilde");
        nuFissionXS = HDF5ToKokkosView<DualView2D>(CMFDCoarseMesh.getDataSet("nu-fission XS"), "nuFissionXS");
        pastFlux = HDF5ToKokkosView<DualView2D>(CMFDCoarseMesh.getDataSet("flux"), "pastFlux");
        removalXS = HDF5ToKokkosView<DualView2D>(CMFDCoarseMesh.getDataSet("removal XS"), "removalXS");
        surf2Cell = HDF5ToKokkosView<DualViewSurf2Cell>(CMFDCoarseMesh.getDataSet("surf2cell"), "surf2Cell");
        scatteringXS = HDF5ToKokkosView<DualView3D>(CMFDCoarseMesh.getDataSet("scattering XS"), "scatteringXS");

        size_t firstCell, lastCell;
        CMFDCoarseMesh.getDataSet("first cell").read(firstCell);
        CMFDCoarseMesh.getDataSet("last cell").read(lastCell);
        nCells = lastCell - firstCell + 1;

        nSurfaces = surf2Cell.extent(0);

        CMFDCoarseMesh.getDataSet("energy groups").read(nGroups);
    };

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
