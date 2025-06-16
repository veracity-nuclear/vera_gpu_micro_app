#pragma once

#include "gtest/gtest.h"
#include "petscvec_kokkos.hpp"
#include "petscmat_kokkos.hpp"
#include "petscksp.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include "highfive/H5File.hpp"

#include <string>
#include <vector>

// PetscCall can't be used in the body of a TEST
#define PetscCallG(call) ASSERT_EQ(call, PETSC_SUCCESS) << "PETSc call failed: " << #call;

/**
 * @brief Test fixture for initializing and finalizing PETSc
 * This allows tests to be run separately or all at once
 */
class PetscKokkosTestEnvironment : public ::testing::Environment
{
private:
  int argc_;
  char **argv_;

public:
  PetscKokkosTestEnvironment(int argc, char **argv) : argc_(argc), argv_(argv) {}

protected:
  void SetUp() override
  {
    Kokkos::initialize();
    PetscCallG(PetscInitialize(&argc_, &argv_, NULL, NULL));
    PetscCallG(PetscLogDefaultBegin());
  }

  void TearDown() override
  {
    PetscCallG(PetscFinalize());
    Kokkos::finalize();
  }
};

void compare2DViewAndVector(
    const Kokkos::View<PetscScalar **, Kokkos::HostSpace> &view,
    const std::vector<std::vector<PetscScalar>> &vec,
    const std::string &message = "");

void compare2DHostAndDevice(
    const Kokkos::View<PetscScalar **, Kokkos::HostSpace> &h_view,
    const Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> &d_view,
    const std::string &message = "");

// Convert a HighFive group (data sets are rows) to a 2D vector
std::vector<std::vector<PetscScalar>> readMatrixFromHDF5(const HighFive::Group &AMatH5);

// Create a PETSc vector (passed by ref) from a std::vector<PetscScalar>
PetscErrorCode createPetscVec(const std::vector<PetscScalar> &vec, Vec &vecPetsc);

// Create a PETSc matrix (passed by ref) from a std::vector<std::vector<PetscScalar>>
PetscErrorCode createPetscMat(const std::vector<std::vector<PetscScalar>> &vecOfVec, Mat &matPetsc);