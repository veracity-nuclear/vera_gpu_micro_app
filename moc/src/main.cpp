#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "argument_parser.hpp"
#include "eigen_solver.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";

    Kokkos::parallel_for("hello", 10, KOKKOS_LAMBDA(int i) {
      printf("Hello from i = %d\n", i);
    });
  }

  // Create argument parser
  ArgumentParser parser(argv[0], "VERA GPU Micro-App for eigenvalue calculations");

  // Add required positional arguments
  parser.add_argument("filename", "Input geometry file");
  parser.add_argument("xs_file", "Cross-section data file");

  // Add some optional arguments (examples)
  parser.add_option("threads", "Number of threads to use", "0");
  parser.add_flag("verbose", "Enable verbose output");
  parser.add_option("device", "Device to use (serial, openmp, cuda)", "serial", {"serial", "openmp", "cuda"});

  // Parse arguments
  if (!parser.parse(argc, argv)) {
      Kokkos::finalize();
      return 1;
  }

  // Get a vector of arguments compatible with the original EigenSolver constructor
  std::vector<std::string> solver_args = parser.get_args(argv[0]);

  // Optional: Access specific arguments if needed
  bool verbose = parser.get_flag("verbose");
  if (verbose) {
      std::cout << "Input file: " << parser.get_positional(0) << std::endl;
      std::cout << "XS file: " << parser.get_positional(1) << std::endl;
  }

  EigenSolver solver(solver_args);
  solver.solve();
  std::cout << "Final keff: " << solver.keff() << std::endl;

  Kokkos::finalize();
  return 0;
}
