#include "PetscKokkosTestEnvironment.hpp"
#include "PetscEigenSolver.hpp"

class CoarseVsFineEigenvalueTest : public ::testing::TestWithParam<std::string> {
protected:
    using Assembler = CSRMatrixAssembler;
    using AssemblySpace = Assembler::AssemblySpace;
    using CoarseData = CMFDData<AssemblySpace>;
    using FineData = FineMeshData<AssemblySpace>;

    void SetUp() override {
        fileName = GetParam();
        HighFive::File file(fileName, HighFive::File::ReadOnly);
        HighFive::Group coarseGroup = file.getGroup("CMFD_CoarseMesh");
        HighFive::Group fineGroup = file.getGroup("CMFD_FineMesh");

        HighFive::DataSet kGoldH5 = file.getDataSet("STATE_0001/keff");
        kGoldH5.read(kGold);

        CoarseData originalCoarseData(coarseGroup);
        FineData fineData(fineGroup);


        CoarseData generatedCoarseData = originalCoarseData;
        fineData.homogenizeAll(generatedCoarseData);

        Assembler mpactAssembler(std::move(originalCoarseData));
        Assembler fineAssembler(std::move(generatedCoarseData));

        fromMPACT = std::make_unique<PetscEigenSolver>(std::make_unique<Assembler>(std::move(mpactAssembler)));
        fromFine = std::make_unique<PetscEigenSolver>(std::make_unique<Assembler>(std::move(fineAssembler)));

        ::testing::TestWithParam<std::string>::SetUp();
    }

    void TearDown() override {
        fromFine.reset();
        fromMPACT.reset();
        ::testing::TestWithParam<std::string>::TearDown();
    }

    std::string fileName;
    double kGold;
    std::unique_ptr<PetscEigenSolver> fromMPACT, fromFine;
};

TEST_P(CoarseVsFineEigenvalueTest, EigenvalueComparison) {
    double kFromMPACT, kFromFine;

    std::cout << "Solving coarse mesh from MPACT coarse data..." << std::endl;
    fromMPACT->solve();
    kFromMPACT = fromMPACT->keff;

    std::cout << "Solving coarse mesh from fine data..." << std::endl;
    fromFine->solve();
    kFromFine = fromFine->keff;

    EXPECT_NEAR(kFromMPACT, kGold, 1e-7) << "Error in MPACT coarse mesh solution";
    // TODO (#64): The tolerances are currently too high.
    EXPECT_NEAR(kFromFine, kFromMPACT, 7e-4) << "Data from fine mesh did not have the same eigenvalue as that from the MPACT coarse mesh.";
}

INSTANTIATE_TEST_SUITE_P(
    CoarseVsFineEigenvalueTests,
    CoarseVsFineEigenvalueTest,
    ::testing::Values(
        "data/pin_7g_16a_3p_serial.h5",
        "data/2x2_7g_16a_3p_serial.h5",
        "data/7x7_7g_16a_3p_serial.h5",
        "data/small_parallel.h5",
        "data/mini-core_7g_16a_3p_serial.h5"
    ));