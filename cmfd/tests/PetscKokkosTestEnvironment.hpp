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

#include "PetscMatrixAssembler.hpp"

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

template <typename DataType, typename ExecutionSpace>
void compare2DViews(
    const Kokkos::View<DataType **, ExecutionSpace> &view1,
    const Kokkos::View<DataType **, ExecutionSpace> &view2,
    const PetscScalar &rtol = 1e-10,
    const PetscScalar &atol = 1e-10,
    const std::string &message = "")
{
  ASSERT_EQ(view1.extent(0), view2.extent(0));
  ASSERT_EQ(view1.extent(1), view2.extent(1));

  auto d_view1 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), view1);
  auto d_view2 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), view2);

  for (size_t i = 0; i < d_view1.extent(0); ++i)
  {
    for (size_t j = 0; j < d_view1.extent(1); ++j)
    {
      const auto val1 = d_view1(i, j);
      const auto val2 = d_view2(i, j);

      ASSERT_NEAR(val1, val2, atol) << "Absolute Error @ (" << i << ", " << j << "): " << message;
      ASSERT_LE((val1 - val2) / val2, rtol) << "Relative Error @ (" << i << ", " << j << "): " << message;
    }
  }
}

template<typename DataType, typename ExecutionSpace>
void compare3DViews(
  const Kokkos::View<DataType ***, ExecutionSpace> &view1,
  const Kokkos::View<DataType ***, ExecutionSpace> &view2,
  const PetscScalar &rtol = 1e-10,
  const PetscScalar &atol = 1e-10,
  const std::string &message = "")
{
  ASSERT_EQ(view1.extent(0), view2.extent(0));
  ASSERT_EQ(view1.extent(1), view2.extent(1));
  ASSERT_EQ(view1.extent(2), view2.extent(2));

  auto d_view1 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), view1);
  auto d_view2 = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), view2);

  for (size_t i = 0; i < d_view1.extent(0); ++i)
  {
    for (size_t j = 0; j < d_view1.extent(1); ++j)
    {
      for (size_t k = 0; k < d_view1.extent(2); ++k)
      {
        const auto val1 = d_view1(i, j, k);
        const auto val2 = d_view2(i, j, k);

        ASSERT_NEAR(val1, val2, atol) << "Absolute Error @ (" << i << ", " << j << ", " << k << "): " << message;
        if (val1 != 0.0 && val2 != 0.0)
          ASSERT_LE((val1 - val2) / val2, rtol) << "Relative Error @ (" << i << ", " << j << ", " << k << "): " << message;
      }
    }
  }
}

void vectorsAreParallel(
    const Vec &v1,
    const Vec &v2,
    PetscScalar tol = 1.e-7);

// Convert a HighFive group (data sets are rows) to a 2D vector
std::vector<std::vector<PetscScalar>> readMatrixFromHDF5(const HighFive::Group &AMatH5);

// Create a PETSc vector (passed by ref) from a std::vector<PetscScalar>
PetscErrorCode createPetscVec(const std::vector<PetscScalar> &vec, Vec &vecPetsc);

// Create a PETSc matrix (passed by ref) from a std::vector<std::vector<PetscScalar>>
PetscErrorCode createPetscMat(const std::vector<std::vector<PetscScalar>> &vecOfVec, Mat &matPetsc);

// Helper function to detect if T derives from any PetscMatrixAssembler<Space>
template <typename T, typename = void>
struct isPetscMatrixAssembler
{
  template <typename AnySpace>
  static std::true_type test(const PetscMatrixAssembler<AnySpace> *);
  static std::false_type test(...);
  static constexpr bool value = decltype(test(std::declval<T *>()))::value;
};

// Struct for working with PetscMatrixAssembler without actually assembling a matrix
// and instead using the matrices and vectors from an HDF5 file
struct DummyMatrixAssembler : public PetscMatrixAssembler<Kokkos::DefaultHostExecutionSpace>
{
  using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
  using AssemblyMemorySpace = Kokkos::HostSpace;

  Vec fluxGold;
  double kGold;
  size_t nGroups, nCells;

  DummyMatrixAssembler() = default;
  DummyMatrixAssembler(const HighFive::File &file);
  ~DummyMatrixAssembler()
  {
    PetscCallCXXAbort(PETSC_COMM_SELF, VecDestroy(&fluxGold));
  };

  PetscErrorCode _assembleM() override
  {
    // No-op, MMat is already initialized in the constructor
    return PETSC_SUCCESS;
  }
  PetscErrorCode _assembleFission(const FluxView &flux) override
  {
    // No-op, we don't need to assemble fission in this dummy assembler
    return PETSC_SUCCESS;
  }
};