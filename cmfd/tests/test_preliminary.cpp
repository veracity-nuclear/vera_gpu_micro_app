#include "gtest/gtest.h"
#include "petscmat.h"
#include "highfive/H5File.hpp"
#include "highfive/H5DataSet.hpp"
#include "highfive/H5DataSpace.hpp"

#include <string>
#include <vector>

// PetscCall can't be used in the body of a TEST
#define PetscCallG(call) ASSERT_EQ(call, PETSC_SUCCESS) << "PETSc call failed: " << #call;

/**
 * Step 01: Can we read HDF5 files?
 * Step 02: Can we use PETSc and read data?
 * Step 03: Can we use Kokkos with PETSc?
 */

/**
 * @brief Test fixture for initializing and finalizing PETSc
 * This allows tests to be run separately or all at once
 */
class CMFDPrelim : public ::testing::Environment{
protected:
  void SetUp() override {
    PetscCallG(PetscInitializeNoArguments());
  }

  void TearDown() override {
    PetscCallG(PetscFinalize());
  }
};

std::vector<std::vector<PetscScalar>> readMatrixFromHDF5(const HighFive::Group& AMatH5) {
  std::vector<std::vector<PetscScalar>> AMatLocal;
  std::vector<PetscScalar> rowData;

  const std::vector<std::string> rowNames = AMatH5.listObjectNames();

  // assuming rows are in the correct order...
  for (const std::string& row_name : rowNames) {
      HighFive::DataSet rowDataset = AMatH5.getDataSet(row_name);
      rowDataset.read(rowData);
      AMatLocal.emplace_back(rowData);
  }

  return AMatLocal;
}

PetscErrorCode createPetscVec(const std::vector<PetscScalar>& vec, Vec& vecPetsc) {
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

PetscErrorCode createPetscMat(const std::vector<std::vector<PetscScalar>>& vecOfVec, Mat& matPetsc) {
  PetscInt nRows = vecOfVec.size();
  PetscInt nCols = nRows; // Assuming square matrix for simplicity

  PetscFunctionBeginUser;
  
  PetscCall(MatCreate(PETSC_COMM_WORLD, &matPetsc));
  PetscCall(MatSetSizes(matPetsc, PETSC_DECIDE, PETSC_DECIDE, nRows, nCols));
  PetscCall(MatSetFromOptions(matPetsc));

  for (PetscInt i = 0; i < nRows; ++i) {
    for (PetscInt j = 0; j < nCols; ++j) {
      if (vecOfVec[i][j] != 0.0) {
        PetscCall(MatSetValue(matPetsc, i, j, vecOfVec[i][j], INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(matPetsc, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(matPetsc, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Test if a HDF5 file can be opened and read
TEST(s01_hdf5, readVector) {
  std::vector<PetscScalar> bVecLocal;
  std::string filename = "data/pin_7g_16a_3p_serial.h5";

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");
  bVecH5.read(bVecLocal);

  ASSERT_FALSE(bVecLocal.empty()) << "The vector 'b' should not be empty.";    
}

// Check if the matrix 'A' can be read from the HDF5 file
TEST(s01_hdf5, readMatrix) {
  std::string filename = "data/pin_7g_16a_3p_serial.h5";

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");

  std::vector<std::vector<PetscScalar>> AMatLocal = readMatrixFromHDF5(AMatH5);

  ASSERT_FALSE(AMatLocal.empty()) << "The matrix 'A' should not be empty.";
  for (const auto& row : AMatLocal) {
      ASSERT_FALSE(row.empty()) << "Each row of the matrix 'A' should not be empty.";
    }
}

// Test if a vector can be created in PETSc and filled with data from an HDF5 file
TEST(s02_petsc, hdf5ToVector){
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  std::vector<PetscScalar> bVecLocal;
  Vec bVecPetsc;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");
  bVecH5.read(bVecLocal);
  
  PetscCallG(createPetscVec(bVecLocal, bVecPetsc));

  PetscScalar value;
  for (PetscInt i = 0; i < bVecLocal.size(); ++i) {
    PetscCallG(VecGetValues(bVecPetsc, 1, &i, &value));
    ASSERT_EQ(value, bVecLocal[i]) << "Value mismatch at index " << i;
  }

  PetscCallG(VecDestroy(&bVecPetsc));
}

// Test if a matrix can be created in PETSc and filled with data from an HDF5 file
TEST(s02_petsc, hdf5ToMatrix){
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  Mat AMatPetsc;
  PetscScalar dummyValue;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");

  std::vector<std::vector<PetscScalar>> AMatLocal = readMatrixFromHDF5(AMatH5);

  PetscCallG(createPetscMat(AMatLocal, AMatPetsc));
  PetscInt nRows = AMatLocal.size();

  // Verify the matrix
  for (size_t i = 0; i < nRows; ++i) {
    for (size_t j = 0; j < nRows; ++j) {
      PetscCallG(MatGetValue(AMatPetsc,i, j, &dummyValue));
      ASSERT_EQ(dummyValue, AMatLocal[i][j]) << "Value mismatch at (" << i << ", " << j << ")";
    }
  }

  PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_STDOUT_WORLD));
  // PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_DRAW_WORLD)); // need x11

  PetscCallG(MatDestroy(&AMatPetsc));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CMFDPrelim());
  return RUN_ALL_TESTS();
}