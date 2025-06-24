#include "PetscEigenSolver.hpp"
#include "PetscKokkosTestEnvironment.hpp"

using AssemblerPtr = PetscEigenSolver::AssemblerPtr;
using AssemblerPtrFactory = std::function<AssemblerPtr(const HighFive::Group&)>;
using Params = std::tuple<std::string, AssemblerPtrFactory>;

class PetscEigenSolverTest : public :: testing::TestWithParam<Params>
{
public:
    std::string filePath;
    std::unique_ptr<PetscEigenSolver> solver;
    DummyMatrixAssembler dummyAssembler;

    void SetUp() override
    {
        filePath = std::get<0>(GetParam());
        HighFive::File file(filePath, HighFive::File::ReadOnly);
        HighFive::Group coarseMeshData = file.getGroup("CMFD_CoarseMesh");

        dummyAssembler = DummyMatrixAssembler(file);

        AssemblerPtrFactory factory = std::get<1>(GetParam());
        solver = std::make_unique<PetscEigenSolver>(factory(coarseMeshData));

        ::testing::TestWithParam<Params>::SetUp();
    }

    void TearDown() override
    {
        // Other object lifetimes should handle all PETSc objects
        ::testing::TestWithParam<Params>::TearDown();
    }

    void solveOneIteration()
    {
        PetscCallG(solver->solve(1));

        std::vector<double> keffHistory = solver->keffHistory;
        ASSERT_EQ(keffHistory.size(), 1) << "Keff history should have one entry after one iteration";

        ASSERT_NE(keffHistory[0], keffHistory[1]) << "Keff should change after one iteration";
    }

    void solveTwoIterations()
    {
        PetscCallG(solver->solve(2));

        std::vector<double> keffHistory = solver->keffHistory;
        ASSERT_EQ(keffHistory.size(), 2) << "Keff history should have two entries after two iterations";

        ASSERT_NE(keffHistory[0], keffHistory[1]) << "Keff should change after first iteration";
        ASSERT_NE(keffHistory[1], keffHistory[0]) << "Keff should change after second iteration";
    }
};

TEST_P(PetscEigenSolverTest, TestOneIteration)
{
    // Test a single iterationd
    solveOneIteration();
}

TEST_P(PetscEigenSolverTest, TestTwoIterations)
{
    // Test two iterations
    solveTwoIterations();
}

template<typename AssemblerType>
std::vector<Params> createParams(const std::vector<std::string>& files)
{
    std::vector<Params> params;

    AssemblerPtrFactory assemblerPtrFactory = [](const HighFive::Group &coarseMeshData)
    {
        return std::make_unique<AssemblerType>(coarseMeshData);
    };

    for (const auto& file : files) {
        params.emplace_back(file, assemblerPtrFactory);
    }

    return params;
}

static const std::vector<std::string> testFiles = {
    "data/pin_7g_16a_3p_serial.h5",
    "data/7x7_7g_16a_3p_serial.h5"
};

INSTANTIATE_TEST_SUITE_P(
    TestEigenSimple,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<SimpleMatrixAssembler>(testFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestEigenCOO,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<COOMatrixAssembler>(testFiles))
);
