#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "argument_parser.hpp"
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "kokkos_moc.hpp"
#include "eigen_solver.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  // Create argument parser with pre-configured arguments
  ArgumentParser parser = ArgumentParser::vera_gpu_moc_parser(argv[0]);

  // Parse arguments
  if (!parser.parse(argc, argv)) {
      Kokkos::finalize();
      return 1;

  }

  // Get the verbosity
  bool verbose = parser.get_flag("verbose");

  {
    std::cout << "Kokkos execution spaces enabled:\n";
#ifdef KOKKOS_ENABLE_SERIAL
    std::cout << "  - SERIAL\n";
    if (verbose) {
      Kokkos::parallel_for("serial_hello", Kokkos::RangePolicy<Kokkos::Serial>(0, 10), KOKKOS_LAMBDA(int i) {
        printf("Hello from parallel_for test loop index i = %d/10\n", i);
      });
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "  - OPENMP\n";
    if (verbose) {
      Kokkos::parallel_for("openmp_hello", Kokkos::RangePolicy<Kokkos::OpenMP>(0, 10), KOKKOS_LAMBDA(int i) {
        printf("Hello from parallel_for test loop index i = %d/10\n", i);
      });
    }
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    std::cout << "  - OPENMPTARGET\n";
    if (verbose) {
      Kokkos::parallel_for("openmptarget_hello", Kokkos::RangePolicy<Kokkos::OpenMPTarget>(0, 10), KOKKOS_LAMBDA(int i) {
        printf("Hello from parallel_for test loop index i = %d/10\n", i);
      });
    }
#endif
#ifdef KOKKOS_ENABLE_THREADS
    std::cout << "  - THREADS\n";
    if (verbose) {
      Kokkos::parallel_for("threads_hello", Kokkos::RangePolicy<Kokkos::Threads>(0, 10), KOKKOS_LAMBDA(int i) {
        printf("Hello from parallel_for test loop index i = %d/10\n", i);
      });
    }
#endif
#ifdef KOKKOS_ENABLE_CUDA
    std::cout << "  - CUDA\n";
    if (verbose) {
      Kokkos::parallel_for("cuda_hello", Kokkos::RangePolicy<Kokkos::Cuda>(0, 10), KOKKOS_LAMBDA(int i) {
        printf("Hello from parallel_for test loop index i = %d/10\n", i);
      });
    }
#endif
#ifdef KOKKOS_ENABLE_HIP
    std::cout << "  - HIP\n";
    if (verbose) {
      Kokkos::parallel_for("hip_hello", Kokkos::RangePolicy<Kokkos::HIP>(0, 10), KOKKOS_LAMBDA(int i) {
        printf("Hello from parallel_for test loop index i = %d/10\n", i);
      });
    }
#endif
#ifdef KOKKOS_ENABLE_SYCL
    std::cout << "  - SYCL\n";
    if (verbose) {
      Kokkos::parallel_for("sycl_hello", Kokkos::RangePolicy<Kokkos::SYCL>(0, 10), KOKKOS_LAMBDA(int i) {
        printf("Hello from parallel_for test loop index i = %d/10\n", i);
      });
    }
#endif
  }

  // Read the sweeper argument
  std::string sweeper_type = parser.get_option("sweeper");

  // Print all the argument options
  if (verbose) {
      // Print positional arguments
      std::cout << "Input file: " << parser.get_positional(0) << std::endl;
      std::cout << "XS file: " << parser.get_positional(1) << std::endl;

      // Print optional arguments
      std::cout << "Threads: " << parser.get_option("threads") << std::endl;
      std::cout << "Verbose: " << (parser.get_flag("verbose") ? "true" : "false") << std::endl;
      std::cout << "Sweeper: " << parser.get_option("sweeper") << std::endl;
      std::cout << "Device: " << parser.get_option("device") << std::endl;
  } else {
      if (sweeper_type == "kokkos") {
          std::string device = parser.get_option("device");
          std::cout << "Selected kokkos backend: " << device << std::endl;
      }
  }

  // Get a vector of arguments compatible with the original EigenSolver constructor
  std::vector<std::string> solver_args = parser.get_args(argv[0]);

  // Set up the sweeper
  BaseMOC* sweeper;
  if (sweeper_type == "kokkos") {
    sweeper = new KokkosMOC(parser);
  } else if (sweeper_type == "serial") {
    sweeper = new SerialMOC(parser);
  }

  // Pass by reference to EigenSolver
  EigenSolver eigen_solver(solver_args, sweeper);

  // Use the eigen_solver
  eigen_solver.solve();
  std::cout << "Final keff: " << eigen_solver.keff() << std::endl;

  Kokkos::finalize();
  return 0;
}
