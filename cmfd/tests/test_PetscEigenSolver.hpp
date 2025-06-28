#pragma once

#include "PetscEigenSolver.hpp"
#include "PetscKokkosTestEnvironment.hpp"

using AssemblerPtr = PetscEigenSolver::AssemblerPtr;
using AssemblerPtrFactory = std::function<AssemblerPtr(const HighFive::Group&)>;
using Params = std::tuple<std::string, AssemblerPtrFactory>;

// Creates Params for the PetscEigenSolverTest where the template determines the type of Assembler used
// and the vector of strings (argument) contains the file paths to be tested
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
        ASSERT_EQ(keffHistory.size(), 2) << "Keff history should have two entries after one iteration";
        ASSERT_NE(keffHistory[0], keffHistory[1]) << "Keff should change after one iteration";
    }

    void solve()
    {
        PetscCallG(solver->solve(1000));

        ASSERT_NEAR(solver->keff, dummyAssemblerPtr->kGold, solver->tol)
            << "Keff = " << solver->keff
            << ", expected = " << dummyAssemblerPtr->kGold
            << ", tolerance = " << solver->tol;

        vectorsAreParallel(solver->currentFlux, dummyAssemblerPtr->fluxGold, solver->tol);
    }
};

TEST_P(PetscEigenSolverTest, TestOneIteration)
{
    // Test a single iteration
    solveOneIteration();
}

TEST_P(PetscEigenSolverTest, FullSolve)
{
    // Test the full solve
    solve();
}