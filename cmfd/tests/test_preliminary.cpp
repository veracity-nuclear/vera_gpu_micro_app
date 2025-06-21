/*
  This file tests and demonstrates key functionalities of PETSc and Kokkos
  needed for the CMFD code.
*/
#include "PetscKokkosTestEnvironment.hpp"

// Create a PETSc vector (passed by ref) of type VECKOKKOS
PetscErrorCode createPetscVecKokkos(const std::vector<PetscScalar> &vec, Vec &vecPetsc)
{
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
PetscErrorCode createPetscMatKokkos(const std::vector<std::vector<PetscScalar>> &vecOfVec, Mat &matPetsc)
{
  PetscInt nRows = vecOfVec.size();
  PetscInt nCols = nRows;

  PetscFunctionBeginUser;

  PetscInt numNonZero = 0;
  for (PetscInt i = 0; i < nRows; ++i)
  {
    for (PetscInt j = 0; j < nCols; ++j)
    {
      if (vecOfVec[i][j] != 0.0)
      {
        numNonZero++;
      }
    }
  }

  // There are a lot of options here for splitting up the matrix into submatrices per mpi rank
  //  (on the PETSc side), i.e., # of rows/cols per rank and number of zeros on and off the diagonal (per row).
  PetscCall(MatCreateAIJKokkos(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
                               nRows, nCols,
                               PETSC_DEFAULT, NULL,
                               PETSC_DEFAULT, NULL,
                               &matPetsc));

  if (numNonZero > 0)
  {
    std::vector<PetscInt> rowIndices, colIndices;
    std::vector<PetscScalar> values;
    rowIndices.reserve(numNonZero);
    colIndices.reserve(numNonZero);
    values.reserve(numNonZero);

    for (PetscInt i = 0; i < nRows; ++i)
    {
      for (PetscInt j = 0; j < nCols; ++j)
      {
        if (vecOfVec[i][j] != 0.0)
        {
          rowIndices.emplace_back(i);
          colIndices.emplace_back(j);
          values.emplace_back(vecOfVec[i][j]);
        }
      }
    }

    /* // Option A: Do not use Kokkos views (This works)
    PetscCall(MatSetPreallocationCOO(matPetsc, numNonZero, rowIndices.data(), colIndices.data()));
    PetscCall(MatSetValuesCOO(matPetsc, values.data(), INSERT_VALUES));
    */

    // Option B: Use Kokkos views on the host (This works)
    Kokkos::View<PetscInt *, Kokkos::DefaultHostExecutionSpace> h_rowIndices("rowIndicesKokkos", numNonZero);
    Kokkos::View<PetscInt *, Kokkos::DefaultHostExecutionSpace> h_colIndices("colIndicesKokkos", numNonZero);
    Kokkos::View<PetscScalar *, Kokkos::DefaultHostExecutionSpace> h_values("valuesKokkos", numNonZero);
    for (PetscInt i = 0; i < numNonZero; ++i)
    {
      h_rowIndices(i) = rowIndices[i];
      h_colIndices(i) = colIndices[i];
      h_values(i) = values[i];
    }
    PetscCall(MatSetPreallocationCOO(matPetsc, numNonZero, h_rowIndices.data(), h_colIndices.data()));
    PetscCall(MatSetValuesCOO(matPetsc, h_values.data(), INSERT_VALUES));

    /* // Option C: Use Kokkos views on the device (This does NOT work)
    Kokkos::View<PetscInt *, Kokkos::Cuda> d_rowIndices = Kokkos::create_mirror_view_and_copy(Kokkos::Cuda(), h_rowIndices);
    Kokkos::View<PetscInt *, Kokkos::Cuda> d_colIndices = Kokkos::create_mirror_view_and_copy(Kokkos::Cuda(), h_colIndices);
    Kokkos::View<PetscScalar *, Kokkos::Cuda> d_values = Kokkos::create_mirror_view_and_copy(Kokkos::Cuda(), h_values);
    PetscCall(MatSetPreallocationCOO(matPetsc, numNonZero, d_rowIndices.data(), d_colIndices.data()));
    PetscCall(MatSetValuesCOO(matPetsc, d_values.data(), INSERT_VALUES));
    */

  }
  else
  {
    PetscCall(MatSetPreallocationCOO(matPetsc, 0, NULL, NULL));
  }

  // Don't need to (and shouldn't) call MatAssemblyBegin/End for COO matrices
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Generates dummy data for a Kokkos view using parallel_for, which can't be used in the body of a TEST
void generateDummyKokkosView(Kokkos::View<PetscScalar *> &vecKokkos, const int vecSize = 5)
{
  const PetscScalar number = 2 * M_PI;
  Kokkos::parallel_for("Initialize bVec", vecSize, KOKKOS_LAMBDA(const int i) {
    vecKokkos(i) = Kokkos::sin(static_cast<PetscScalar>(i) / number); // Example data generation
  });

  Kokkos::fence();
}

/**
 * @brief Generates a square tridiagonal matrix in CSR format using Kokkos.
 * @param aValues 1D (nnz) Kokkos view for non-zero values of the matrix.
 * @param iRow 1D (nRows + 1) Kokkos view that encodes the index of aValues where each row starts.
 * @param jCol 1D (nnz) Kokkos view that encodes the column indices of aValues.
 * @param nRows The number of rows (and thus columns) in the matrix.
 *
 * https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)
 */
void generateTridiagKokkosAIJ(Kokkos::View<PetscScalar *> &aValues, Kokkos::View<PetscInt *> &iRow, Kokkos::View<PetscInt *> &jCol, const int nRows)
{
  const PetscScalar float_nRows = static_cast<PetscScalar>(nRows);
  Kokkos::parallel_for("fillTriAIJ", nRows, KOKKOS_LAMBDA(const int rowIdx) {
      const PetscInt diagElementIdx = rowIdx * 3;
      if (rowIdx > 0) {
        iRow(rowIdx) = diagElementIdx - 1;

        // lower diagonal
        aValues(diagElementIdx - 1) = diagElementIdx / float_nRows;
        jCol(diagElementIdx - 1) = rowIdx - 1;
      } else {
        iRow(rowIdx) = 0;

        // We only need to do this once and take advantage of the if else already used
        // This entry does not correspond to rowIdx == 0
        iRow(nRows) = 3 * nRows - 2; // last element is NNZ
      }

      // diagonal
      aValues(diagElementIdx) = Kokkos::sqrt(Kokkos::sinh(diagElementIdx/float_nRows + 1.0));
      jCol(diagElementIdx) = rowIdx;

      // upper diagonal
      if (rowIdx < nRows - 1) {
        aValues(diagElementIdx + 1) = 1;
        jCol(diagElementIdx + 1) = rowIdx + 1;
      } });
  Kokkos::fence();
}

// Test if a HDF5 file can be opened and read
TEST(s01_hdf5, readVector)
{
  std::vector<PetscScalar> bVecLocal;
  std::string filename = "data/pin_7g_16a_3p_serial.h5";

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");
  bVecH5.read(bVecLocal);

  ASSERT_FALSE(bVecLocal.empty()) << "The vector 'b' should not be empty.";
}

// Check if the matrix 'A' can be read from the HDF5 file
TEST(s01_hdf5, readMatrix)
{
  std::string filename = "data/pin_7g_16a_3p_serial.h5";

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");

  std::vector<std::vector<PetscScalar>> AMatLocal = readMatrixFromHDF5(AMatH5);

  ASSERT_FALSE(AMatLocal.empty()) << "The matrix 'A' should not be empty.";
  for (const auto &row : AMatLocal)
  {
    ASSERT_FALSE(row.empty()) << "Each row of the matrix 'A' should not be empty.";
  }
}

TEST(s01_hdf5, hdf5ToKokkosView)
{
  std::string filename = "data/7x7_7g_16a_3p_serial.h5";
  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::DataSet dataH5 = file.getDataSet("CMFD_CoarseMesh/flux");
  std::vector<size_t> dims = dataH5.getDimensions();

  std::vector<std::vector<PetscScalar>> dataLocal(dims[0], std::vector<PetscScalar>(dims[1], 0.0));
  dataH5.read(dataLocal);

  // You can't read H5 into a Host mirror as the layout may be different
  Kokkos::View<PetscScalar **, Kokkos::HostSpace> h_data(Kokkos::ViewAllocateWithoutInitializing("dataKokkos"), dims[0], dims[1]);
  dataH5.read(h_data.data());

  // Can only read views from HDF5 to Kokkos on the host space
  Kokkos::View<PetscScalar**> d_data(Kokkos::ViewAllocateWithoutInitializing("device data"), dims[0], dims[1]);
  Kokkos::View<PetscScalar**>::HostMirror h_mirrorData = Kokkos::create_mirror_view(d_data);
  Kokkos::deep_copy(h_mirrorData, h_data); // Changes the layout to the device layout
  // At this point we can deallocate h_data (Kokkos uses RAII)

  // Check if the data matches the local data
  for (size_t i = 0; i < dims[0]; ++i)
  {
    for (size_t j = 0; j < dims[1]; ++j)
    {
      ASSERT_EQ(h_mirrorData(i, j), dataLocal[i][j]) << "Error reading data from H5: ";
      // printf("dataKokkos(%zu, %zu) = %f, dataLocal(%zu, %zu) = %f\n", i, j, dataKokkos(i, j), i, j, dataLocal[i][j]);
    }
  }

  // See if we can copy the data to the device and back
  Kokkos::deep_copy(d_data, h_mirrorData);

  Kokkos::View<PetscScalar**>::HostMirror h_dataCheck = Kokkos::create_mirror_view(d_data);
  Kokkos::deep_copy(h_dataCheck, d_data);

  for (size_t i = 0; i < dims[0]; ++i)
  {
    for (size_t j = 0; j < dims[1]; ++j)
    {
      ASSERT_EQ(h_dataCheck(i, j), dataLocal[i][j]) << "Error copying data to device and back: ";
    }
  }

  // Testing dual view
  using DualView = Kokkos::DualView<PetscScalar **>;
  DualView dualData(Kokkos::ViewAllocateWithoutInitializing("dual"), dims[0], dims[1]);

  DualView::t_host h_dualData = dualData.view_host();
  Kokkos::deep_copy(h_dualData, h_data);
  dualData.template modify<typename DualView::host_mirror_space>();

  for (size_t i = 0; i < dims[0]; ++i)
  {
    for (size_t j = 0; j < dims[1]; ++j)
    {
      ASSERT_EQ(h_dualData(i, j), dataLocal[i][j]) << "Error reading data into device dual view: ";
    }
  }

  dualData.template sync<typename Kokkos::DefaultExecutionSpace>();
  DualView::t_dev d_dualData = dualData.view_device();

  Kokkos::View<PetscScalar**>::HostMirror h_dualDataCheck = Kokkos::create_mirror_view(h_dualData);
  Kokkos::deep_copy(h_dualDataCheck, d_dualData);
  for (size_t i = 0; i < dims[0]; ++i)
  {
    for (size_t j = 0; j < dims[1]; ++j)
    {
      ASSERT_EQ(h_dualDataCheck(i, j), dataLocal[i][j]) << "Error copying data to device dual view and back: ";
    }
  }
}

// Test if a vector can be created in PETSc and filled with data from an HDF5 file
TEST(s02_petsc, hdf5ToVector)
{
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  std::vector<PetscScalar> bVecLocal;
  Vec bVecPetsc;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");
  bVecH5.read(bVecLocal);

  PetscCallG(createPetscVec(bVecLocal, bVecPetsc));

  PetscScalar value;
  for (PetscInt i = 0; i < bVecLocal.size(); ++i)
  {
    PetscCallG(VecGetValues(bVecPetsc, 1, &i, &value));
    ASSERT_EQ(value, bVecLocal[i]) << "Value mismatch at index " << i;
  }

  PetscCallG(VecDestroy(&bVecPetsc));
}

// Test if a matrix can be created in PETSc and filled with data from an HDF5 file
TEST(s02_petsc, hdf5ToMatrix)
{
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
  for (size_t i = 0; i < nRows; ++i)
  {
    for (size_t j = 0; j < nRows; ++j)
    {
      PetscCallG(MatGetValue(AMatPetsc, i, j, &dummyValue));
      ASSERT_EQ(dummyValue, AMatLocal[i][j]) << "Value mismatch at (" << i << ", " << j << ")";
    }
  }

  PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_STDOUT_WORLD));
#ifndef NDEBUG
  PetscCallG(PetscOptionsSetValue(NULL, "-draw_size", "1920,1080")); // Set the size of the draw window
  PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_DRAW_WORLD));           // need x11 but allows visualization
#endif

  PetscCallG(MatDestroy(&AMatPetsc));
}

// Test if a linear system can be solved using PETSc with data from an HDF5 file
TEST(s02_petsc, solve)
{
  // See petsc/src/ksp/ksp/tutorials/ex1.c for reference
  double tol = 1.e-7;
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  std::vector<PetscScalar> bVecLocal, xVecLocalGold;
  Mat AMatPetsc;
  Vec bVecPetsc, xVecPetsc;
  KSP ksp; // Linear solver context
  PC pc;   // Preconditioner context
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
  PetscCallG(VecSetType(bVecPetsc, VECKOKKOS));

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

  for (PetscInt i = 0; i < xVecLocalGold.size(); i++)
  {
    PetscCallG(VecGetValues(xVecPetsc, 1, &i, &value));
    EXPECT_NEAR(value, xVecLocalGold[i], tol) << "Value mismatch at index " << i;
  }

  PetscCallG(KSPDestroy(&ksp));
  PetscCallG(MatDestroy(&AMatPetsc));
  PetscCallG(VecDestroy(&xVecPetsc));
  PetscCallG(VecDestroy(&bVecPetsc));
}

// Create a PETSc vector of type VECKOKKOS
TEST(s03_kokkos, petscKokkosVec)
{
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  std::vector<PetscScalar> bVecLocal;
  Vec bVecPetsc;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::DataSet bVecH5 = file.getDataSet("CMFD_Matrix/b");
  bVecH5.read(bVecLocal);

  PetscCallG(createPetscVecKokkos(bVecLocal, bVecPetsc));

  PetscScalar value;
  for (PetscInt i = 0; i < bVecLocal.size(); ++i)
  {
    PetscCallG(VecGetValues(bVecPetsc, 1, &i, &value));
    ASSERT_EQ(value, bVecLocal[i]) << "Value mismatch at index " << i;
  }

  PetscCallG(VecDestroy(&bVecPetsc));
}

// Create a PETSc matrix of type MATAIJKOKKOS
TEST(s03_kokkos, petscKokkosMat)
{
  std::string filename = "data/pin_7g_16a_3p_serial.h5";
  Mat AMatPetsc;
  PetscScalar dummyValue;

  PetscFunctionBeginUser;

  HighFive::File file(filename, HighFive::File::ReadOnly);
  HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");

  std::vector<std::vector<PetscScalar>> AMatLocal = readMatrixFromHDF5(AMatH5);

  PetscCallG(createPetscMatKokkos(AMatLocal, AMatPetsc));
  PetscInt nRows = AMatLocal.size();

  for (size_t i = 0; i < nRows; ++i)
  {
    for (size_t j = 0; j < nRows; ++j)
    {
      PetscCallG(MatGetValue(AMatPetsc, i, j, &dummyValue));
      ASSERT_EQ(dummyValue, AMatLocal[i][j]) << "Value mismatch at (" << i << ", " << j << ")";
    }
  }

  PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_STDOUT_WORLD));

  PetscCallG(MatDestroy(&AMatPetsc));
}

// Solve a linear system using PETSc with Kokkos vectors and matrices
TEST(s03_kokkos, solveKokkos)
{
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

  for (PetscInt i = 0; i < xVecLocalGold.size(); i++)
  {
    PetscCallG(VecGetValues(xVecPetsc, 1, &i, &value));
    EXPECT_NEAR(value, xVecLocalGold[i], 10 * tol) << "Value mismatch at index " << i;
  }

  PetscCallG(KSPDestroy(&ksp));
  PetscCallG(MatDestroy(&AMatPetsc));
  PetscCallG(VecDestroy(&xVecPetsc));
  PetscCallG(VecDestroy(&bVecPetsc));
}

// Test if a Kokkos view can be converted to a PETSc vectors
TEST(s03_kokkos, kokkosViewToPetscVec)
{
  const size_t vectorLength = 5;
  Vec bVecPetsc;
  Kokkos::View<PetscScalar *> bVecKokkos("bVec", vectorLength);

  std::cout << "bVec memory space: " << typeid(decltype(bVecKokkos)::memory_space).name() << std::endl;

  PetscFunctionBeginUser;

  generateDummyKokkosView(bVecKokkos, vectorLength); // You can't use Kokkos Lambdas in a TEST...

  // Create a PETSc vector from the Kokkos view
  PetscCallG(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF, 1, vectorLength, bVecKokkos.data(), &bVecPetsc));

  // Get the Kokkos view from the PETSc vector (must be const data type in view)
  Kokkos::View<const PetscScalar *, Kokkos::DefaultExecutionSpace::memory_space> d_values;
  PetscCallG(VecGetKokkosView(bVecPetsc, &d_values));

  // Copy the values to the host from the device so we can inspect it
  // Kokkos::View<PetscScalar*, Kokkos::HostSpace::memory_space> h_values("bVecHost", vectorLength);
  Kokkos::View<PetscScalar *, Kokkos::HostSpace::memory_space>::HostMirror h_values = Kokkos::create_mirror_view(d_values);

  Kokkos::deep_copy(h_values, d_values);
  Kokkos::fence(); // Ensure the copy is complete before accessing

  PetscScalar value;
  for (size_t i = 0; i < vectorLength; ++i)
  {
    value = h_values(i);
    std::cout << "bVec[" << i << "] = " << value << std::endl;

    // These range of values are determined generateKokkosView,
    //  but the purpose of these tests is to make sure data are valid
    ASSERT_GT(value, -1.0) << "Value should be greater than -1. The data may not have been initialized correctly.";
    ASSERT_LE(value, 1.0) << "Value should be less than or equal to one. The data may not have been initialized correctly.";
  }

  PetscCallG(VecRestoreKokkosView(bVecPetsc, &d_values));

  // You can also view the vector using PETSc's built-in viewer,
  //  but calculating the norm (what is done in other tests to
  //  verify data integrity) doesn't work with this vector for some reason
  PetscCallG(VecView(bVecPetsc, PETSC_VIEWER_STDOUT_WORLD));

  PetscCallG(VecDestroy(&bVecPetsc));
}

// Test if we can use an A, I, and J Kokkos view to create a tridiagonal PETSc matrix
TEST(s03_kokkos, kokkosViewToPetscMat)
{
  const PetscInt numRows = 5;
  const PetscInt numNonZero = 3 * numRows - 2; // 3 non-zero elements per row, except for the first and last rows
  Mat AMatPetsc;
  Kokkos::View<PetscScalar *> aValues("aValues", numNonZero);
  Kokkos::View<PetscInt *> iRow("iRow", numRows + 1);
  Kokkos::View<PetscInt *> jCol("jCol", numNonZero);

  PetscFunctionBeginUser;

  generateTridiagKokkosAIJ(aValues, iRow, jCol, numRows);

  // The NULL allows you to specify the number of non-zero elements per row,
  //  which doesn't make a big difference for a tridiagonal matrix
  PetscCallG(MatCreateSeqAIJKokkos(PETSC_COMM_SELF, numRows, numRows, numNonZero, NULL, &AMatPetsc));
  PetscCallG(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, numRows, numRows, iRow, jCol, aValues, &AMatPetsc));

  PetscCallG(MatView(AMatPetsc, PETSC_VIEWER_STDOUT_WORLD));

  // Calculate norm of the matrix to verify it is non-zero
  PetscReal norm;
  PetscCallG(MatNorm(AMatPetsc, NORM_FROBENIUS, &norm));
  ASSERT_FALSE(norm == 0.0) << "The matrix norm is 0, likely garbage data";
  ASSERT_FALSE(std::isnan(norm)) << "The matrix norm is NaN, likely garbage data";
  ASSERT_FALSE(std::isinf(norm)) << "The matrix norm is Inf, likely garbage data";

  // Don't destroy AMatPetsc as Kokkos::finalize will do it for us
}

// Test if we can solve a linear system with PETSc initialized with Kokkos views
TEST(s03_kokkos, solveFromViews)
{
  const PetscScalar tol = 1.e-7;
  const PetscInt numRows = 100;
  const PetscInt numNonZero = 3 * numRows - 2;

  Vec bVecPetsc, xVecPetsc;
  Mat AMatPetsc;
  KSP ksp;
  PC pc;
  Kokkos::View<PetscScalar *> bVecKokkos("bVec", numRows);
  Kokkos::View<PetscScalar *> aValues("aValues", numNonZero);
  Kokkos::View<PetscInt *> iRow("iRow", numRows + 1);
  Kokkos::View<PetscInt *> jCol("jCol", numNonZero);

  PetscFunctionBeginUser;

  generateDummyKokkosView(bVecKokkos, numRows);
  PetscCallG(VecCreateSeqKokkosWithArray(PETSC_COMM_SELF, 1, numRows, bVecKokkos.data(), &bVecPetsc));
  PetscCallG(VecDuplicate(bVecPetsc, &xVecPetsc));

  generateTridiagKokkosAIJ(aValues, iRow, jCol, numRows);
  PetscCallG(MatCreateSeqAIJKokkos(PETSC_COMM_SELF, numRows, numRows, numNonZero, NULL, &AMatPetsc));
  PetscCallG(MatCreateSeqAIJKokkosWithKokkosViews(PETSC_COMM_SELF, numRows, numRows, iRow, jCol, aValues, &AMatPetsc));

  PetscCallG(KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCallG(KSPSetOperators(ksp, AMatPetsc, AMatPetsc));
  PetscCallG(KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));

  PetscCallG(KSPGetPC(ksp, &pc));
  PetscCallG(PCSetType(pc, PCJACOBI));

  PetscCallG(KSPSolve(ksp, bVecPetsc, xVecPetsc));

  if constexpr(numRows <= 100)
  {
    PetscCallG(VecView(xVecPetsc, PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscReal norm;
  PetscCallG(VecNorm(xVecPetsc, NORM_2, &norm));
  ASSERT_FALSE(norm == 0.0) << "The solution vector norm is 0, likely garbage data";
  ASSERT_FALSE(std::isnan(norm)) << "The solution vector norm is NaN, likely garbage data";
  ASSERT_FALSE(std::isinf(norm)) << "The solution vector norm is Inf, likely garbage data";

  PetscCallG(KSPDestroy(&ksp));
  PetscCallG(VecDestroy(&xVecPetsc));
  PetscCallG(VecDestroy(&bVecPetsc));
}