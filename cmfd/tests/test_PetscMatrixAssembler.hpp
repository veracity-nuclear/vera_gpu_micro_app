/*
  This file tests if the PetscMatrixAssembler works as intended.
*/
#pragma once

#include "PetscKokkosTestEnvironment.hpp"
#include "PetscMatrixAssembler.hpp"

class PetscMatrixAssemblerTest : public ::testing::TestWithParam<std::string>
{
public:
  using AssemblerPtr = std::unique_ptr<MatrixAssemblerInterface>;

  std::string filePath;
  std::unique_ptr<HighFive::Group> coarseMeshData;
  size_t nCells, nGroups;
  double pastKeff;
  Mat goldMat;
  Vec goldVec;

  void SetUp() override
  {
    filePath = GetParam();

    HighFive::File file(filePath, HighFive::File::ReadOnly);

    DummyMatrixAssembler dummyAssembler(file);

    // I'm not super happy about copying values but this is for performant code.
    MatDuplicate(dummyAssembler.MMat, MAT_COPY_VALUES, &goldMat);

    // You can't duplicate (the type) of a vector and copy (the values) at the same time...
    VecDuplicate(dummyAssembler.fissionVec, &goldVec);
    VecCopy(dummyAssembler.fissionVec, goldVec);

    pastKeff = dummyAssembler.kGold;
    nCells = dummyAssembler.nCells;
    nGroups = dummyAssembler.nGroups;

    HighFive::Group _coarseMeshData = file.getGroup("CMFD_CoarseMesh");
    coarseMeshData = std::make_unique<HighFive::Group>(_coarseMeshData);

    ::testing::TestWithParam<std::string>::SetUp();
  }

  void TearDown() override
  {
    PetscCallG(MatDestroy(&goldMat));
    PetscCallG(VecDestroy(&goldVec));
    ::testing::TestWithParam<std::string>::TearDown();
  }


  void compareMatrices(const AssemblerPtr& assemblerPtr, PetscReal tolerance = 1.0e-10) const
  {
    Mat testMat = assemblerPtr->getM();

    PetscInt testRows, testCols, goldRows, goldCols;
    PetscCallG(MatGetSize(testMat, &testRows, &testCols));
    PetscCallG(MatGetSize(goldMat, &goldRows, &goldCols));
    ASSERT_EQ(testRows, goldRows) << "Row dimensions of testMat and goldMat do not match";
    ASSERT_EQ(testCols, goldCols) << "Column dimensions of testMat and goldMat do not match";

    Mat diffMat;
    PetscCallG(MatDuplicate(testMat, MAT_COPY_VALUES, &diffMat));

    // MatAXPY computes Y = a*X + Y (but the function signature is (Y, a, X, sparsityPattern))
    // diffMat = testMat - goldMat (we filled diffMat with testMat)
    PetscCallG(MatAXPY(diffMat, -1.0, goldMat, DIFFERENT_NONZERO_PATTERN));
    // Note, these matrices should have the same sparsity pattern, but if we use
    // SAME_NONZERO_PATTERN, we will get a segfault that isn't super descriptive
    // in the case that testMat and goldMat don't have the same sparsity pattern

    PetscReal norm;
    PetscCallG(MatNorm(diffMat, NORM_FROBENIUS, &norm));


    std::cout << "Matrix norm difference: " << norm << std::endl;

    if (norm > tolerance)
    {
      PetscCallG(PetscOptionsSetValue(NULL, "-draw_size", "1920,1080")); // Set the size of the draw window
      MatView(diffMat, PETSC_VIEWER_DRAW_WORLD);

      MatView(diffMat, PETSC_VIEWER_STDOUT_WORLD);

      Mat& mat = diffMat;
      Mat mat = goldMat;
      PetscInt ncols, nrows;
      PetscCallG(MatGetSize(mat, &nrows, &ncols));

      static constexpr PetscInt _max = 100;
      for (PetscInt i = 0; i < std::min(_max, nrows); ++i)
      {
        const PetscInt *cols;
        const PetscScalar *vals;
        PetscInt ncols_row;
        PetscCallG(MatGetRow(mat, i, &ncols_row, &cols, &vals));
        std::cout << "Row " << i << ": ";
        for (PetscInt j = 0; j < ncols_row; ++j)
        {
          std::cout << "(" << cols[j] << ", " << vals[j] << ") ";
        }
        std::cout << std::endl;
        PetscCallG(MatRestoreRow(mat, i, &ncols_row, &cols, &vals));
      }
    }

    ASSERT_LE(norm, tolerance) << "Matrices differ by " << norm << " (tolerance: " << tolerance << ")";

    PetscCallG(MatDestroy(&diffMat));
  }

  void compareVectors(AssemblerPtr& assemblerPtr, PetscReal tolerance = 1.0e-7) const
  {
    std::vector<PetscScalar> pastFlux1D(nCells * nGroups);
    std::vector<std::vector<PetscScalar>> pastFlux2D;
    Vec pastFluxVec, testFissionVec, diffVec;

    // Get 1D Flux Vec to use as an input to assembler.getFissionSource()
    HighFive::DataSet pastFluxH5 = coarseMeshData->getDataSet("flux");
    pastFluxH5.read(pastFlux2D);

    for (size_t cellIdx = 0; cellIdx < nCells; ++cellIdx)
    {
      for (size_t groupIdx = 0; groupIdx < nGroups; ++groupIdx)
      {
        pastFlux1D[cellIdx * nGroups + groupIdx] = pastFlux2D[groupIdx][cellIdx];
      }
    }

    createPetscVec(pastFlux1D, pastFluxVec);
    VecSetType(pastFluxVec, VECKOKKOS);

    testFissionVec = assemblerPtr->getFissionSource(pastFluxVec);

    // Divide the testFissionVec by pastKeff to compare with the goldVec
    VecScale(testFissionVec, 1.0 / pastKeff);

    PetscInt testSize, goldSize;
    PetscCallG(VecGetSize(testFissionVec, &testSize));
    PetscCallG(VecGetSize(goldVec, &goldSize));
    ASSERT_EQ(testSize, goldSize) << "Vector sizes do not match";

    PetscCallG(VecDuplicate(testFissionVec, &diffVec));
    PetscCallG(VecCopy(testFissionVec, diffVec));
    // VecAXPY computes Y = a*X + Y (but the function signature is (Y, a, X))
    // diffVec = testFissionVec - goldVec (we filled diffVec with testFissionVec)
    PetscCallG(VecAXPY(diffVec, -1.0, goldVec));

    PetscReal norm;
    PetscCallG(VecNorm(diffVec, NORM_2, &norm));
    std::cout << "Vector norm difference: " << norm << std::endl;
    if (norm > tolerance)
    {
      for (PetscInt i = 0; i < testSize; ++i)
      {
        PetscScalar diffValue, goldValue, testValue;
        PetscCallG(VecGetValues(diffVec, 1, &i, &diffValue));
        PetscCallG(VecGetValues(goldVec, 1, &i, &goldValue));
        PetscCallG(VecGetValues(testFissionVec, 1, &i, &testValue));
        if (std::fabs(diffValue) > tolerance)
        {
          std::cerr << "Test/Gold(" << i << ") = " << testValue / goldValue << std::endl;
        }
        std::cerr << "The above values should be close to 1.0" << std::endl;
      }
      PetscCallG(PetscOptionsSetValue(NULL, "-draw_size", "1920,1080")); // Set the size of the draw window
      VecView(diffVec, PETSC_VIEWER_DRAW_WORLD);
    }
    ASSERT_LE(norm, tolerance) << "Vectors differ by " << norm << " (tolerance: " << tolerance << ")";
    PetscCallG(VecDestroy(&diffVec));

  }

  // I could have templated the above methods instead, but using a unique pointer
  // to the interface is how these classes are designed to work.
  // The interface only exposes the methods I want, so it acts almost like protected inheritance.
  template<typename AssemblerType>
  AssemblerPtr createAssemblerPtr() const
  {
    static_assert(isPetscMatrixAssembler<AssemblerType>::value,
      "AssemblerType must be a PetscMatrixAssembler or derived from one");
    return std::make_unique<AssemblerType>(*coarseMeshData);
  }

  template<typename AssemblerType>
  void compareMatrices(PetscReal tolerance = 1.0e-10) const
  {
    AssemblerPtr assembler = createAssemblerPtr<AssemblerType>();
    compareMatrices(assembler, tolerance);
  }

  template<typename AssemblerType>
  void compareVectors(PetscReal tolerance = 1.0e-7) const
  {
    AssemblerPtr assembler = createAssemblerPtr<AssemblerType>();
    compareVectors(assembler, tolerance);
  }

};

TEST_P(PetscMatrixAssemblerTest, TestSimpleMatrixAssembler)
{
  compareMatrices<SimpleMatrixAssembler>();
}

TEST_P(PetscMatrixAssemblerTest, TestSimpleVectorAssembler)
{
  compareVectors<SimpleMatrixAssembler>();
}

TEST_P(PetscMatrixAssemblerTest, TestCOOMatrixAssembler)
{
  compareMatrices<COOMatrixAssembler>();
}

TEST_P(PetscMatrixAssemblerTest, TestCOOVectorAssembler)
{
  compareVectors<COOMatrixAssembler>();
}

TEST_P(PetscMatrixAssemblerTest, TestCSRMatrixAssembler)
{
  compareMatrices<CSRMatrixAssembler>();
}

TEST_P(PetscMatrixAssemblerTest, TestCSRVectorAssembler)
{
  compareVectors<CSRMatrixAssembler>();
}