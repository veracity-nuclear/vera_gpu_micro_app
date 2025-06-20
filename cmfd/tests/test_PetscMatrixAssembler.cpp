/*
  This file tests if the PetscMatrixAssembler works as intended.
*/
#include "PetscKokkosTestEnvironment.hpp"
#include "PetscMatrixAssembler.hpp"

class PetscMatrixAssemblerTest : public ::testing::TestWithParam<std::string>
{
public:
  // Helper function to detect if T derives from any PetscMatrixAssembler<Space>
  template <typename T, typename = void>
  struct isPetscMatrixAssembler
  {
    template <typename AnySpace>
    static std::true_type test(const PetscMatrixAssembler<AnySpace> *);
    static std::false_type test(...);
    static constexpr bool value = decltype(test(std::declval<T *>()))::value;
  };

  double pastKeff;
  std::string filePath;
  std::unique_ptr<HighFive::Group> coarseMeshData;
  Mat goldMat;
  Vec goldVec;

  void SetUp() override
  {
    filePath = GetParam();

    HighFive::File file(filePath, HighFive::File::ReadOnly);

    DummyMatrixAssembler dummyAssembler(file);
    MatDuplicate(dummyAssembler.A, MAT_COPY_VALUES, &goldMat);
    VecDuplicate(dummyAssembler.b, &goldVec);
    VecCopy(dummyAssembler.b, goldVec);
    pastKeff = dummyAssembler.kGold;

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

  template<typename AssemblerType>
  AssemblerType createAssembler() const
  {
    static_assert(isPetscMatrixAssembler<AssemblerType>::value,
      "AssemblerType must be a PetscMatrixAssembler or derived from one");
    return AssemblerType(*coarseMeshData);
  }

  void compareMatrices(const Mat& testMat, PetscReal tolerance = 1.0e-10) const
  {
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
    }

    ASSERT_LE(norm, tolerance) << "Matrices differ by " << norm << " (tolerance: " << tolerance << ")";

    PetscCallG(MatDestroy(&diffMat));
  }

  template<typename AssemblerType>
  void compareMatrices(PetscReal tolerance = 1.0e-10) const
  {
    AssemblerType assembler = createAssembler<AssemblerType>();

    Mat testMat = assembler.assembleM();
    compareMatrices(testMat, tolerance);
    PetscCallG(MatDestroy(&testMat));
  }

  template<typename AssemblerType>
  void compareVectors(PetscReal tolerance = 1.0e-7) const
  {
    using AssemblySpace = typename AssemblerType::AssemblySpace;
    using View2D = typename AssemblerType::CMFDDataType::View2D;
    using FluxView = typename AssemblerType::FluxView;

    AssemblerType assembler = createAssembler<AssemblerType>();

    size_t nCells = assembler.cmfdData.nCells;
    size_t nGroups = assembler.cmfdData.nGroups;

    // Convert to 1D
    View2D pastFlux2D = HDF5ToKokkosView<View2D>(coarseMeshData->getDataSet("flux"), "flux");
    FluxView pastFlux("pastFlux1D", nCells * nGroups);
    Kokkos::parallel_for("flux2Dto1D", Kokkos::MDRangePolicy<AssemblySpace, Kokkos::Rank<2>>({0, 0}, {pastFlux2D.extent(0), pastFlux2D.extent(1)}),
      KOKKOS_LAMBDA(const PetscInt groupIdx, const PetscInt cellIdx)
      {
        pastFlux(cellIdx * nGroups + groupIdx) = pastFlux2D(groupIdx, cellIdx);
      });

    Vec testVec = assembler.assembleF(pastFlux);

    // Divide the testVec by pastKeff to compare with the goldVec
    VecScale(testVec, 1.0 / pastKeff);

    PetscInt testSize, goldSize;
    PetscCallG(VecGetSize(testVec, &testSize));
    PetscCallG(VecGetSize(goldVec, &goldSize));
    ASSERT_EQ(testSize, goldSize) << "Vector sizes do not match";

    Vec diffVec;
    PetscCallG(VecDuplicate(testVec, &diffVec));
    PetscCallG(VecCopy(testVec, diffVec));
    // VecAXPY computes Y = a*X + Y (but the function signature is (Y, a, X))
    // diffVec = testVec - goldVec (we filled diffVec with testVec)
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
        PetscCallG(VecGetValues(testVec, 1, &i, &testValue));
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

INSTANTIATE_TEST_SUITE_P(
  TestAssemblyLight,
  PetscMatrixAssemblerTest,
  ::testing::Values(
    "data/pin_7g_16a_3p_serial.h5",
    "data/7x7_7g_16a_3p_serial.h5"));

INSTANTIATE_TEST_SUITE_P(
  TestAssemblyHeavy,
  PetscMatrixAssemblerTest,
  ::testing::Values(
    "data/mini-core_7g_16a_3p_serial.h5"));

