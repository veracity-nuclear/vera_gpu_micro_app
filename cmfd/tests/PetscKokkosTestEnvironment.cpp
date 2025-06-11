#include "PetscKokkosTestEnvironment.hpp"

void compare2DViewAndVector(
    const Kokkos::View<PetscScalar **, Kokkos::HostSpace> &view,
    const std::vector<std::vector<PetscScalar>> &vec,
    const std::string &message)
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

void compare2DHostAndDevice(
    const Kokkos::View<PetscScalar **, Kokkos::HostSpace> &h_view,
    const Kokkos::View<PetscScalar **, Kokkos::DefaultExecutionSpace> &d_view,
    const std::string &message)
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

// Convert a HighFive group (data sets are rows) to a 2D vector
std::vector<std::vector<PetscScalar>> readMatrixFromHDF5(const HighFive::Group &AMatH5)
{
  const std::vector<std::string> rowNames = AMatH5.listObjectNames();
  const size_t n = rowNames.size(); // Square matrix dimension

  std::vector<std::vector<PetscScalar>> AMatLocal(n, std::vector<PetscScalar>(n));

  for (size_t i = 0; i < n; ++i)
  {
    HighFive::DataSet rowDataset = AMatH5.getDataSet(rowNames[i]);
    rowDataset.read(AMatLocal[i]);
  }

  return AMatLocal;
}

// Create a PETSc vector (passed by ref) from a std::vector<PetscScalar>
PetscErrorCode createPetscVec(const std::vector<PetscScalar> &vec, Vec &vecPetsc)
{
  PetscInt n = vec.size();

  PetscFunctionBeginUser;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &vecPetsc));
  PetscCall(VecSetSizes(vecPetsc, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(vecPetsc));

  std::vector<PetscInt> indices(n);
  std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., size-1

  PetscCall(VecSetValues(vecPetsc, n, indices.data(), vec.data(), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(vecPetsc));
  PetscCall(VecAssemblyEnd(vecPetsc));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create a PETSc matrix (passed by ref) from a std::vector<std::vector<PetscScalar>>
PetscErrorCode createPetscMat(const std::vector<std::vector<PetscScalar>> &vecOfVec, Mat &matPetsc)
{
  PetscInt nRows = vecOfVec.size();
  PetscInt nCols = nRows; // Assuming square matrix for simplicity

  PetscFunctionBeginUser;

  PetscCall(MatCreate(PETSC_COMM_WORLD, &matPetsc));
  PetscCall(MatSetSizes(matPetsc, PETSC_DECIDE, PETSC_DECIDE, nRows, nCols));
  PetscCall(MatSetFromOptions(matPetsc));

  for (PetscInt i = 0; i < nRows; ++i)
  {
    for (PetscInt j = 0; j < nCols; ++j)
    {
      if (vecOfVec[i][j] != 0.0)
      {
        PetscCall(MatSetValue(matPetsc, i, j, vecOfVec[i][j], INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(matPetsc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(matPetsc, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}