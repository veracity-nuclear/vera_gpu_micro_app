#include "argument_parser.hpp"
#include "PetscEigenSolver.hpp"

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

    std::unique_ptr<PetscEigenSolver> solver;
    solver = std::make_unique<PetscEigenSolver>(filename);

    PetscCall(solver->solve());

    solver.reset(); // Ensure solver is destroyed before finalizing PETSc
    PetscCall(PetscFinalize());
  }
  Kokkos::finalize();
  return 0;
}