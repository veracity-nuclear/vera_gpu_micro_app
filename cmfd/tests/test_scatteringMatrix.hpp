#pragma once

#include "ScatteringMatrix.hpp"
#include "hdf5_kokkos.hpp"
#include "PetscKokkosTestEnvironment.hpp"

class ScatteringMatrixTest : public :: testing::TestWithParam<std::string>
{
public:
    using AssemblySpace = Kokkos::DefaultHostExecutionSpace;
    using ScatteringMatrixType = ScatteringMatrix<AssemblySpace>;
    using View1D = typename ScatteringMatrixType::View1D;
    using View2D = typename ScatteringMatrixType::View2D;
    using View3D = typename ScatteringMatrixType::View3D;

    std::string filePath;

    View1D vals;
    std::vector<std::vector<std::vector<double>>> trueScatteringMat;
    std::unique_ptr<ScatteringMatrixType> scatteringMatrixPtr;

    void SetUp() override
    {
        filePath = GetParam();
        std::string denseFilePath = filePath.substr(0, filePath.size()-3) + "-dense.h5";
        HighFive::File denseFile(denseFilePath, HighFive::File::ReadOnly);
        HighFive::DataSet scatteringXSDataSet = denseFile.getDataSet("CMFD_CoarseMesh/scattering XS");
        scatteringXSDataSet.read(trueScatteringMat);

        HighFive::File sparseFile(filePath, HighFive::File::ReadOnly);
        HighFive::Group scatteringXSGroup = sparseFile.getGroup("CMFD_CoarseMesh/scattering XS");
        scatteringMatrixPtr = std::make_unique<ScatteringMatrixType>(scatteringXSGroup);

        vals = HDF5ToKokkosView<View1D>(scatteringXSGroup.getDataSet("vals"), "vals");

        ::testing::TestWithParam<std::string>::SetUp();
    }

    void TearDown() override
    {
        ::testing::TestWithParam<std::string>::TearDown();
    }

    void testDenseCreation()
    {
        View3D scatteringXS = scatteringMatrixPtr->constructDense(vals);

        ASSERT_EQ(scatteringXS.extent(0), trueScatteringMat.size());
        ASSERT_EQ(scatteringXS.extent(1), trueScatteringMat[0].size());
        ASSERT_EQ(scatteringXS.extent(2), trueScatteringMat[0][0].size());

        for (size_t gFrom = 0; gFrom < scatteringXS.extent(0); ++gFrom)
        {
            for (size_t gTo = 0; gTo < scatteringXS.extent(1); ++gTo)
            {
                for (size_t cell = 0; cell < scatteringXS.extent(2); ++cell)
                {
                    ASSERT_DOUBLE_EQ(scatteringXS(gFrom, gTo, cell), trueScatteringMat[gFrom][gTo][cell]);
                }
            }
        }
    }
};

TEST_P(ScatteringMatrixTest, testDenseCreation)
{
    testDenseCreation();
}