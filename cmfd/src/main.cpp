#include "argument_parser.hpp"
#include "PetscEigenSolver.hpp"
#include "DavidsonEigenSolver.hpp"
#include <string>

ArgumentParser veraGpuCmfdParser(const std::string& program_name) {
    ArgumentParser parser(program_name, "VERA GPU CMFD");

    parser.add_argument("filename", "Input H5 File");

    return parser;
}

int main(int argc, char* argv[]) {

  ArgumentParser parser = veraGpuCmfdParser(argv[0]);
  if (!parser.parse(argc, argv)) {
    return 1;
  }

  std::string filename = parser.get_positional(0);
  std::cout << "Using input file: " << filename << std::endl;

  Kokkos::initialize(argc, argv);
  {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    // -eigensolver davidson (default) | power
    char solverName[64] = "davidson";
    PetscCall(PetscOptionsGetString(nullptr, nullptr, "-eigensolver", solverName, sizeof(solverName), nullptr));
    if (std::string(solverName) == "davidson") {
        auto solver = std::make_unique<DavidsonEigenSolver>(filename);
        PetscCall(solver->solve());
        PetscPrintf(PETSC_COMM_WORLD, "Final keff = %.9f, eigen-residual ||M phi - F phi/k|| / ||M phi|| = %.3e after %zu iterations\n",
                    solver->keff, solver->residualHistory.back(), solver->keffHistory.size() - 1);
        solver.reset(); // Ensure solver is destroyed before finalizing PETSc
    } else {
        auto solver = std::make_unique<PetscEigenSolver>(filename);
        PetscCall(solver->solve());
        PetscReal relres;
        PetscCall(solver->eigenResidual(relres));
        PetscPrintf(PETSC_COMM_WORLD, "Final keff = %.9f, eigen-residual ||M phi - F phi/k|| / ||M phi|| = %.3e after %zu iterations\n",
                    solver->keff, relres, solver->keffHistory.size() - 1);
        solver.reset();
    }
    PetscCall(PetscFinalize());
  }
  Kokkos::finalize();
  return 0;
}