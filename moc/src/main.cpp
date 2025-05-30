#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "eigen_solver.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";

    Kokkos::parallel_for("hello", 10, KOKKOS_LAMBDA(int i) {
      printf("Hello from i = %d\n", i);
    });
  }

  const std::vector<std::string> args(argv, argv + argc);
  if (args.size() != 3) {
      std::cerr << "Usage: " << args[0] << " <filename> <XS file>" << std::endl;
      Kokkos::finalize();
      return 1;
  }

  EigenSolver solver(args);
  solver.solve();
  std::cout << "Final keff: " << solver.keff() << std::endl;

  Kokkos::finalize();
  return 0;
}
