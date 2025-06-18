/*
  This file tests if the PetscMatrixAssembler works as intended.

  Notes: I'm not the biggest fan of the current implementation.
  - We redundantly create the gold matrix for each AssemblerType
  - The current test fixture "parameterizes" the assembler type, which incentives
    the test file as the variable used for comparison, but I think it would be better
    to refactor this at some point such that the parameter is the file name and the
    test fixture creates all three assemblers. I think this approach is better for
    comparing assembly times across methods, but it would be a little tricky to implement
    as clean as I'd like.
*/
#include "PetscKokkosTestEnvironment.hpp"
#include "PetscMatrixAssembler.hpp"

template <typename AssemblerType>
class PetscMatrixAssemblerTest : public ::testing::Test
{
protected:
  // Helper functions to detect if T derives from any PetscMatrixAssembler<Space>
  template <typename T, typename = void>
  struct isPetscMatrixAssembler
  {
    template <typename AnySpace>
    static std::true_type test(const PetscMatrixAssembler<AnySpace> *);
    static std::false_type test(...);
    static constexpr bool value = decltype(test(std::declval<T *>()))::value;
  };

  static_assert(isPetscMatrixAssembler<AssemblerType>::value,
                "AssemblerType must be a PetscMatrixAssembler or derived from one");

  bool isSetup = false;
  Mat goldMat, testMat;
  AssemblerType assembler;

  void setupWithFile(const std::string &filePath)
  {
    isSetup = true;

    HighFive::File file(filePath, HighFive::File::ReadOnly);

    HighFive::Group AMatH5 = file.getGroup("CMFD_Matrix/A");
    std::vector<std::vector<PetscScalar>> goldMatVec = readMatrixFromHDF5(AMatH5);
    PetscCallG(createPetscMat(goldMatVec, goldMat));

    HighFive::Group cmfdGroup = file.getGroup("CMFD_CoarseMesh");
    assembler = AssemblerType(cmfdGroup);

    const std::string stageName = "Assemble:" + std::string(typeid(AssemblerType).name()) + filePath;

    PetscLogStage stage;
    PetscLogStageRegister(stageName.c_str(), &stage);

    PetscLogStagePush(stage);
    testMat = assembler.assemble();
    PetscLogStagePop();
  }

  void TearDown() override
  {
    PetscCallG(MatDestroy(&goldMat));
    PetscCallG(MatDestroy(&testMat));
    ::testing::Test::TearDown();
  }

  void compareMatrices(PetscReal tolerance = 1.0e-10)
  {
    ASSERT_TRUE(isSetup) << "Test environment not set up. Call setupWithFile() before running tests.";

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

      // PetscScalar diffValue, goldValue, testValue;
      // for (PetscInt i = 0; i < nRows; ++i)
      // {
      //   for (PetscInt j = 0; j < nCols; ++j)
      //   {
      //     PetscCallG(MatGetValue(diffMat, i, j, &diffValue));
      //     if (std::fabs(diffValue) > tolerance)
      //     {
      //       PetscCallG(MatGetValue(goldMat, i, j, &goldValue));
      //       PetscCallG(MatGetValue(testMat, i, j, &testValue));
      //       printf("Row %d, Col %d: gold %f, test %f, diff %f\n", i, j, goldValue, testValue, diffValue);
      //     }
      //   }
      // }

      PetscCallG(PetscOptionsSetValue(NULL, "-draw_size", "1920,1080")); // Set the size of the draw window
      MatView(diffMat, PETSC_VIEWER_DRAW_WORLD);
    }

    ASSERT_LE(norm, tolerance) << "Matrices differ by " << norm << " (tolerance: " << tolerance << ")";

    PetscCallG(MatDestroy(&diffMat));
  }
};

using AssemblerTypes = ::testing::Types<
    SimpleMatrixAssembler
    , COOMatrixAssembler
    // , KokkosMatrixAssembler
    >;

TYPED_TEST_SUITE(PetscMatrixAssemblerTest, AssemblerTypes);

TYPED_TEST(PetscMatrixAssemblerTest, AssembleAndCompare)
{
  const std::vector<std::string> testFiles = {
      "data/pin_7g_16a_3p_serial.h5",
      "data/7x7_7g_16a_3p_serial.h5",
      "data/mini-core_7g_16a_3p_serial.h5"
  };

  for (const auto &file : testFiles)
  {
    this->setupWithFile(file);
    this->compareMatrices();
  }
}
