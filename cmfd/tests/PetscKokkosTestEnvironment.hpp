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

inline void compare2DViewAndVector(
    const Kokkos::View<PetscScalar **, Kokkos::HostSpace> &view,
    const std::vector<std::vector<PetscScalar>> &vec,
    const std::string &message = "")
{
  ASSERT_EQ(view.extent(0), vec.size()) << message;
  ASSERT_EQ(view.extent(1), vec[0].size()) << message;

  for (size_t i = 0; i < view.extent(0); ++i)
  {
    for (size_t j = 0; j < view.extent(1); ++j)
    {
      ASSERT_EQ(view(i, j), vec[i][j]) << message;
    }
  }
}

inline void compare2DHostAndDevice(
    const Kokkos::View<PetscScalar **, Kokkos::HostSpace> &h_view,
    const Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> &d_view,
    const std::string &message = "")
{
  ASSERT_EQ(h_view.extent(0), d_view.extent(0)) << message;
  ASSERT_EQ(h_view.extent(1), d_view.extent(1)) << message;

  Kokkos::View<PetscScalar **>::HostMirror h_viewCheck = Kokkos::create_mirror_view(d_view);
  Kokkos::deep_copy(h_viewCheck, d_view);

  for (size_t i = 0; i < h_view.extent(0); ++i)
  {
    for (size_t j = 0; j < h_view.extent(1); ++j)
    {
      ASSERT_DOUBLE_EQ(h_view(i, j), h_viewCheck(i, j)) << message;
    }
  }
}