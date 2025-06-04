#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "eigen_solver.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos execution spaces enabled:\n";
#ifdef KOKKOS_ENABLE_SERIAL
    std::cout << "  - SERIAL\n";
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "  - OPENMP\n";
#endif
#ifdef KOKKOS_ENABLE_THREADS
    std::cout << "  - THREADS\n";
#endif
#ifdef KOKKOS_ENABLE_CUDA
    std::cout << "  - CUDA\n";
#endif
#ifdef KOKKOS_ENABLE_HIP
    std::cout << "  - HIP\n";
#endif
#ifdef KOKKOS_ENABLE_SYCL
    std::cout << "  - SYCL\n";
#endif
    std::cout << "Kokkos default backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";

    Kokkos::parallel_for("hello", 10, KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
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
