/*
  This file tests and demonstrates key functionalities of PETSc and Kokkos
  needed for the CMFD code.
*/

#include "gtest/gtest.h"
#include <petscvec_kokkos.hpp>
#include "petscksp.h"
#include "Kokkos_Core.hpp"
#include "highfive/H5File.hpp"

#include <string>
#include <vector>

// PetscCall can't be used in the body of a TEST
#define PetscCallG(call) ASSERT_EQ(call, PETSC_SUCCESS) << "PETSc call failed: " << #call;

/**
 * Step 01: Can we read HDF5 files?
 * Step 02: Can we use PETSc and read data?
 * Step 03: Can we use Kokkos with PETSc?
 * TODO: Look into PETSc DMs
 */

/**
 * @brief Test fixture for initializing and finalizing PETSc
 * This allows tests to be run separately or all at once
 */
class CMFDPrelim : public ::testing::Environment{
private:
  int argc_;
  char** argv_;

public:
  CMFDPrelim(int argc, char** argv) : argc_(argc), argv_(argv) {}

protected:
  void SetUp() override {
    Kokkos::initialize();
    PetscCallG(PetscInitialize(&argc_, &argv_, NULL, NULL));
    PetscCallG(PetscLogDefaultBegin());
  }

  void TearDown() override {
    PetscCallG(PetscFinalize());
    Kokkos::finalize();
  }
};

// Convert a HighFive group (data sets are rows) to a 2D vector
std::vector<std::vector<PetscScalar>> readMatrixFromHDF5(const HighFive::Group& AMatH5) {
  const std::vector<std::string> rowNames = AMatH5.listObjectNames();
  const size_t n = rowNames.size(); // Square matrix dimension

  std::vector<std::vector<PetscScalar>> AMatLocal(n, std::vector<PetscScalar>(n));

  for (size_t i = 0; i < n; ++i) {
    HighFive::DataSet rowDataset = AMatH5.getDataSet(rowNames[i]);
    rowDataset.read(AMatLocal[i]);
  }

  return AMatLocal;
}

// Create a PETSc vector (passed by ref) from a std::vector<PetscScalar>
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

// Create a PETSc matrix (passed by ref) from a std::vector<std::vector<PetscScalar>>
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

// Create a PETSc vector (passed by ref) of type VECKOKKOS
PetscErrorCode createPetscVecKokkos(const std::vector<PetscScalar>& vec, Vec& vecPetsc) {
  PetscInt n = vec.size();
  std::vector<PetscInt> indices(n);

  PetscFunctionBeginUser;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &vecPetsc));
  PetscCall(VecSetType(vecPetsc, VECKOKKOS));
  PetscCall(VecSetSizes(vecPetsc, PETSC_DECIDE, n));

  std::iota(indices.begin(), indices.end(), 0); // Fill indices with 0, 1, ..., size-1
  PetscCall(VecSetPreallocationCOO(vecPetsc, n, indices.data()));
  PetscCall(VecSetValuesCOO(vecPetsc, vec.data(), INSERT_VALUES));

  // Don't need to call VecAssemblyBegin/End for COO vectors

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create a PETSc matrix (passed by ref) of type MATAIJKOKKOS
PetscErrorCode createPetscMatKokkos(const std::vector<std::vector<PetscScalar>>& vecOfVec, Mat& matPetsc) {
  PetscInt nRows = vecOfVec.size();
  PetscInt nCols = nRows;

  PetscFunctionBeginUser;

  PetscInt numNonZero = 0;
  for (PetscInt i = 0; i < nRows; ++i) {
    for (PetscInt j = 0; j < nCols; ++j) {
      if (vecOfVec[i][j] != 0.0) {
        numNonZero++;
      }
    }
  }

  // There are a lot of options here for splitting up the matrix per mpi rank (on the PETSc side),
  // i.e., # of rows/cols per rank and number of zeros on and off the diagonal (per row).
  PetscCall(MatCreateAIJKokkos(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                               nRows, nCols,
                               PETSC_DEFAULT, NULL,
                               PETSC_DEFAULT, NULL,
                               &matPetsc));

  if (numNonZero > 0) {
    std::vector<PetscInt> rowIndices, colIndices;
    std::vector<PetscScalar> values;
    rowIndices.reserve(numNonZero);
    colIndices.reserve(numNonZero);
    values.reserve(numNonZero);

    for (PetscInt i = 0; i < nRows; ++i) {
      for (PetscInt j = 0; j < nCols; ++j) {
        if (vecOfVec[i][j] != 0.0) {
          rowIndices.emplace_back(i);
          colIndices.emplace_back(j);
          values.emplace_back(vecOfVec[i][j]);
        }
      }
    }

    PetscCall(MatSetPreallocationCOO(matPetsc, numNonZero, rowIndices.data(), colIndices.data()));
    PetscCall(MatSetValuesCOO(matPetsc, values.data(), INSERT_VALUES));
  } else {
    PetscCall(MatSetPreallocationCOO(matPetsc, 0, NULL, NULL));
  }

  // Also don't need to call MatAssemblyBegin/End for COO matrices

  PetscFunctionReturn(PETSC_SUCCESS);
}

// Generates dummy data for a Kokkos view using parallel_for, which can't be used in the body of a TEST
void generateKokkosView(Kokkos::View<PetscScalar*>& bVecKokkos, const int vecSize = 5) {
  const PetscScalar number = -3.14;
  Kokkos::parallel_for("Initialize bVec", vecSize, KOKKOS_LAMBDA(const int i) {
    bVecKokkos(i) = Kokkos::exp(static_cast<PetscScalar>(i) / number); // Example data generation
  });

  Kokkos::fence();
}


PetscErrorCode kokkosViewToPetscMat(Mat& AMatPetsc, const int matSize = 5) {
  PetscFunctionBeginUser;

  Kokkos::View<PetscScalar**> AMat("AMat", matSize, matSize);
  std::cout << "AMat memory space: " << typeid(decltype(AMat)::memory_space).name() << std::endl;

  // For AIJ this needs to be different
  // Kokkos::parallel_for("Initialize AMat",
  //   Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {n, n}),
  //   KOKKOS_LAMBDA(const int i, const int j) {
  //   AMat(i, j) = Kokkos::exp(i / n) * j / n;
  //   });
  // PetscCall(MatCreateSeqAIJKokkosWithKokkosViews); // https://petsc.org/release/manualpages/Mat/MatCreateSeqAIJKokkosWithKokkosViews/

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
  // std::string filename = "data/7x7_7g_16a_3p_serial.h5";
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
  PetscCallG(PetscOptionsSetValue(NULL, "-draw_size", "1920,1080")); // Set the size of the draw window
  PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_DRAW_WORLD)); // need x11 but allows visualization

  PetscCallG(MatDestroy(&AMatPetsc));
}

// Test if a linear system can be solved using PETSc with data from an HDF5 file
TEST(s02_petsc, solve){
  // See petsc/src/ksp/ksp/tutorials/ex1.c for reference
  double tol = 1.e-7;
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  std::vector<PetscScalar> bVecLocal, xVecLocalGold;
  Mat AMatPetsc;
  Vec bVecPetsc, xVecPetsc;
  KSP ksp; // Linear solver context
  PC pc; // Preconditioner context
  PetscScalar value;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");
  HighFive::DataSet xVecH5 = file.getDataSet("CMFD_Matrix/x"); // True solution from file
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");

  xVecH5.read(xVecLocalGold);
  bVecH5.read(bVecLocal);

  std::vector<std::vector<PetscScalar>> AMatLocal = readMatrixFromHDF5(AMatH5);

  PetscCallG(createPetscMat(AMatLocal, AMatPetsc));
  PetscCallG(createPetscVec(bVecLocal, bVecPetsc));

  /*
    This doesn't actually duplicate the values from bVecPetsc,
     it just creates a new vector with the same size
  */
  PetscCallG(VecDuplicate(bVecPetsc, &xVecPetsc));

  PetscCallG(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCallG(KSPSetOperators(ksp, AMatPetsc, AMatPetsc));
  PetscCallG(KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));

  PetscCallG(KSPGetPC(ksp, &pc));
  PetscCallG(PCSetType(pc, PCJACOBI));

  PetscCallG(KSPSolve(ksp, bVecPetsc, xVecPetsc));

  for (PetscInt i = 0; i < xVecLocalGold.size(); i++) {
    PetscCallG(VecGetValues(xVecPetsc, 1, &i, &value));
    EXPECT_NEAR(value, xVecLocalGold[i], tol) << "Value mismatch at index " << i;
  }

  PetscCallG(KSPDestroy(&ksp));
  PetscCallG(MatDestroy(&AMatPetsc));
  PetscCallG(VecDestroy(&xVecPetsc));
  PetscCallG(VecDestroy(&bVecPetsc));
}

// Create a PETSc vector of type VECKOKKOS
TEST(s03_kokkos, petscKokkosVec){
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  std::vector<PetscScalar> bVecLocal;
  Vec bVecPetsc;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");
  bVecH5.read(bVecLocal);

  PetscCallG(createPetscVecKokkos(bVecLocal, bVecPetsc));

  PetscScalar value;
  for (PetscInt i = 0; i < bVecLocal.size(); ++i) {
    PetscCallG(VecGetValues(bVecPetsc, 1, &i, &value));
    ASSERT_EQ(value, bVecLocal[i]) << "Value mismatch at index " << i;
  }

  PetscCallG(VecDestroy(&bVecPetsc));
}

// Create a PETSc matrix of type MATAIJKOKKOS
TEST(s03_kokkos, petscKokkosMat){
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  Mat AMatPetsc;
  PetscScalar dummyValue;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");

  std::vector<std::vector<PetscScalar>> AMatLocal = readMatrixFromHDF5(AMatH5);

  PetscCallG(createPetscMatKokkos(AMatLocal, AMatPetsc));
  PetscInt nRows = AMatLocal.size();

  for (size_t i = 0; i < nRows; ++i) {
    for (size_t j = 0; j < nRows; ++j) {
      PetscCallG(MatGetValue(AMatPetsc,i, j, &dummyValue));
      ASSERT_EQ(dummyValue, AMatLocal[i][j]) << "Value mismatch at (" << i << ", " << j << ")";
    }
  }

  PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_STDOUT_WORLD));

  PetscCallG(MatDestroy(&AMatPetsc));
}

// Solve a linear system using PETSc with Kokkos vectors and matrices
TEST(s03_kokkos, solveKokkos){
  double tol = 1.e-7;
  // std::string filename = "data/pin_7g_16a_3p_serial.h5";
  std::string filename = "data/mini-core_7g_16a_3p_serial.h5";
  std::vector<PetscScalar> bVecLocal, xVecLocalGold;
  Mat AMatPetsc;
  Vec bVecPetsc, xVecPetsc;
  KSP ksp;
  PC pc;
  PetscScalar value;
  PetscLogEvent READ_FILE;

  PetscFunctionBeginUser;
  PetscCallG(PetscLogEventRegister("ReadHDF5File", 0, &READ_FILE));

  PetscCallG(PetscLogEventBegin(READ_FILE, 0, 0, 0, 0));
  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");
  HighFive::DataSet xVecH5 = file.getDataSet("CMFD_Matrix/x");
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");

  xVecH5.read(xVecLocalGold);
  bVecH5.read(bVecLocal);

  std::vector<std::vector<PetscScalar>> AMatLocal = readMatrixFromHDF5(AMatH5);
  PetscCallG(PetscLogEventEnd(READ_FILE, 0, 0, 0, 0));

  PetscCallG(createPetscMatKokkos(AMatLocal, AMatPetsc));
  PetscCallG(createPetscVecKokkos(bVecLocal, bVecPetsc));

  PetscCallG(VecDuplicate(bVecPetsc, &xVecPetsc));

  PetscCallG(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCallG(KSPSetOperators(ksp, AMatPetsc, AMatPetsc));
  PetscCallG(KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));

  PetscCallG(KSPGetPC(ksp, &pc));
  PetscCallG(PCSetType(pc, PCJACOBI));

  // This should use the default kokkos execution space
  PetscCallG(KSPSolve(ksp, bVecPetsc, xVecPetsc));

  for (PetscInt i = 0; i < xVecLocalGold.size(); i++) {
    PetscCallG(VecGetValues(xVecPetsc, 1, &i, &value));
    EXPECT_NEAR(value, xVecLocalGold[i], 10*tol) << "Value mismatch at index " << i;
  }

  PetscCallG(KSPDestroy(&ksp));
  PetscCallG(MatDestroy(&AMatPetsc));
  PetscCallG(VecDestroy(&xVecPetsc));
  PetscCallG(VecDestroy(&bVecPetsc));
}

// Test if a Kokkos view can be converted to a PETSc vectors
TEST(s03_kokkos, kokkosViewToPetscVec){
  const size_t vectorLength = 5;
  Vec bVecPetsc;
  Kokkos::View<PetscScalar*> bVecKokkos("bVec", vectorLength);

  std::cout << "bVec memory space: " << typeid(decltype(bVecKokkos)::memory_space).name() << std::endl;

  PetscFunctionBeginUser;

  generateKokkosView(bVecKokkos, vectorLength); // You can't use Kokkos Lambdas in a TEST...

  // Create a PETSc vector from the Kokkos view
  PetscCallG(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF, 1, vectorLength, bVecKokkos.data(), &bVecPetsc));

  // Get the Kokkos view from the PETSc vector (must be const data type in view)
  Kokkos::View<const PetscScalar*, Kokkos::DefaultExecutionSpace::memory_space> d_values;
  PetscCallG(VecGetKokkosView(bVecPetsc, &d_values));

  // Copy the values to the host from the device so we can inspect it
  Kokkos::View<PetscScalar*, Kokkos::HostSpace::memory_space> h_values("bVecHost", vectorLength);
  Kokkos::deep_copy(h_values, d_values);
  Kokkos::fence(); // Ensure the copy is complete before accessing

  PetscScalar value;
  for (size_t i = 0; i < vectorLength; ++i) {
    value = h_values(i);
    std::cout << "bVec[" << i << "] = " << value << std::endl;

    // These range of values are determined generateKokkosView,
    //  but the purpose of these tests is to make sure data are valid
    ASSERT_GT(value, 0.0) << "Value should be greater than zero. The data may not have been initialized correctly.";
    ASSERT_LE(value, 1.0) << "Value should be less than or equal to one. The data may not have been initialized correctly.";
  }

  PetscCallG(VecRestoreKokkosView(bVecPetsc, &d_values));

  PetscCallG(VecDestroy(&bVecPetsc));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CMFDPrelim(argc, argv));
  return RUN_ALL_TESTS();
}