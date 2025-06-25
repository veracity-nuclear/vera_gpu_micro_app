#include "PetscEigenSolver.hpp"
#include "PetscKokkosTestEnvironment.hpp"

using AssemblerPtr = PetscEigenSolver::AssemblerPtr;
using AssemblerPtrFactory = std::function<AssemblerPtr(const HighFive::Group&)>;
using Params = std::tuple<std::string, AssemblerPtrFactory>;

class PetscEigenSolverTest : public :: testing::TestWithParam<Params>
{
public:
    std::string filePath;
    // (Shallow) copies of PETSc objects don't work as I expect,
    // so we use unique_ptr to manage lifetimes.
    std::unique_ptr<PetscEigenSolver> solver;
    std::unique_ptr<DummyMatrixAssembler> dummyAssemblerPtr;

    void SetUp() override
    {
        filePath = std::get<0>(GetParam());
        HighFive::File file(filePath, HighFive::File::ReadOnly);
        HighFive::Group coarseMeshData = file.getGroup("CMFD_CoarseMesh");

        dummyAssemblerPtr = std::make_unique<DummyMatrixAssembler>(file);

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

    void solve()
    {
        Vec diffVec;
        PetscScalar norm;

        PetscCallG(solver->solve(1000));
        ASSERT_NEAR(solver->keff, dummyAssemblerPtr->kGold, solver->tol)
            << "Keff = " << solver->keff
            << ", expected = " << dummyAssemblerPtr->kGold
            << ", tolerance = " << solver->tol;

        PetscCallG(dummyAssemblerPtr->instantiateVec(diffVec));
        PetscCallG(VecCopy(solver->currentFlux, diffVec));
        PetscCallG(VecAXPY(diffVec, -1.0, dummyAssemblerPtr->fluxGold));

        PetscCallG(VecNorm(diffVec, NORM_2, &norm));

        if (norm > solver->tol) {
            VecView(solver->currentFlux, PETSC_VIEWER_STDOUT_SELF);
            VecView(dummyAssemblerPtr->fluxGold, PETSC_VIEWER_STDOUT_SELF);
            VecView(diffVec, PETSC_VIEWER_DRAW_SELF);
        }

        // ASSERT_LE(norm, solver->tol)
        //     << "Norm of the difference between computed and gold flux is too high: "
        //     << norm << " > " << solver->tol;

        PetscCallG(VecDestroy(&diffVec));
    }
};

TEST_P(PetscEigenSolverTest, TestOneIteration)
{
    // Test a single iteration
    solveOneIteration();
}

TEST_P(PetscEigenSolverTest, TestTwoIterations)
{
    // Test two iterations
    solveTwoIterations();
}

TEST_P(PetscEigenSolverTest, FullSolve)
{
    // Test the full solve
    solve();
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

static const std::vector<std::string> lightTestFiles = {
    "data/pin_7g_16a_3p_serial.h5",
    "data/7x7_7g_16a_3p_serial.h5"
};

static const std::vector<std::string> heavyTestFiles = {
    "data/mini-core_7g_16a_3p_serial.h5",
};

INSTANTIATE_TEST_SUITE_P(
    TestEigenSimpleLight,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<SimpleMatrixAssembler>(lightTestFiles))
);

INSTANTIATE_TEST_SUITE_P(
    TestEigenSimpleHeavy,
    PetscEigenSolverTest,
    ::testing::ValuesIn(createParams<SimpleMatrixAssembler>(heavyTestFiles))
);



// TODO
// INSTANTIATE_TEST_SUITE_P(
//     TestEigenCOO,
//     PetscEigenSolverTest,
//     ::testing::ValuesIn(createParams<COOMatrixAssembler>(testFiles))
// );
