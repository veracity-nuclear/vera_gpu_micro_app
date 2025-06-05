#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "argument_parser.hpp"
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "eigen_solver.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::cout << "Kokkos execution spaces enabled:\n";
#ifdef KOKKOS_ENABLE_SERIAL
    std::cout << "  - SERIAL\n";
    Kokkos::parallel_for("serial_hello", Kokkos::RangePolicy<Kokkos::Serial>(0, 10), KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
    });
#endif
#ifdef KOKKOS_ENABLE_OPENMP
    std::cout << "  - OPENMP\n";
    Kokkos::parallel_for("openmp_hello", Kokkos::RangePolicy<Kokkos::OpenMP>(0, 10), KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
    });
#endif
#ifdef KOKKOS_ENABLE_OPENMPTARGET
    std::cout << "  - OPENMPTARGET\n";
    Kokkos::parallel_for("openmptarget_hello", Kokkos::RangePolicy<Kokkos::OpenMPTarget>(0, 10), KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
    });
#endif
#ifdef KOKKOS_ENABLE_THREADS
    std::cout << "  - THREADS\n";
    Kokkos::parallel_for("threads_hello", Kokkos::RangePolicy<Kokkos::Threads>(0, 10), KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
    });
#endif
#ifdef KOKKOS_ENABLE_CUDA
    std::cout << "  - CUDA\n";
    Kokkos::parallel_for("cuda_hello", Kokkos::RangePolicy<Kokkos::Cuda>(0, 10), KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
    });
#endif
#ifdef KOKKOS_ENABLE_HIP
    std::cout << "  - HIP\n";
    Kokkos::parallel_for("hip_hello", Kokkos::RangePolicy<Kokkos::HIP>(0, 10), KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
    });
#endif
#ifdef KOKKOS_ENABLE_SYCL
    std::cout << "  - SYCL\n";
    Kokkos::parallel_for("sycl_hello", Kokkos::RangePolicy<Kokkos::SYCL>(0, 10), KOKKOS_LAMBDA(int i) {
      printf("Hello from parallel_for test loop index i = %d/10\n", i);
    });
#endif
    std::cout << "Kokkos default backend: " << Kokkos::DefaultExecutionSpace::name() << "\n";

  }

  // Create argument parser
  ArgumentParser parser(argv[0], "VERA GPU Micro-App for eigenvalue calculations");

  // Add required positional arguments
  parser.add_argument("filename", "Input geometry file");
  parser.add_argument("xs_file", "Cross-section data file");

  // Add some optional arguments (examples)
  parser.add_option("threads", "Number of threads to use", "0");
  parser.add_flag("verbose", "Enable verbose output");
  parser.add_option("sweeper", "Sweeper type (serial, kokkos)", "serial", {"serial", "kokkos"});
  parser.add_option("device", "Device to use (serial, openmp, cuda)", "serial", {"serial", "openmp", "cuda"});

  // Parse arguments
  if (!parser.parse(argc, argv)) {
      Kokkos::finalize();
      return 1;

  }

  // Get a vector of arguments compatible with the original EigenSolver constructor
  std::vector<std::string> solver_args = parser.get_args(argv[0]);

  // Read the sweeper argument
  std::string sweeper_type = parser.get_option("sweeper");
  BaseMOC* sweeper;
  if (sweeper_type == "kokkos") {
    std::cerr << "Error: Kokkos sweeper is not yet implemented" << std::endl;
    Kokkos::finalize();
    return 1;
  } else if (sweeper_type == "serial") {
    // Initialize a SerialMOC object with a reference of type BaseMOC
    std::cout << "Using serial sweeper" << std::endl;
    sweeper = new SerialMOC(parser.get_positional(0), parser.get_positional(1));
    std::cout << "sweeper flux " << sweeper->scalar_flux().size() << " x "
              << sweeper->scalar_flux()[0].size() << std::endl;
  }

  // Optional: Access specific arguments if needed
  bool verbose = parser.get_flag("verbose");
  if (verbose) {
      std::cout << "Input file: " << parser.get_positional(0) << std::endl;
      std::cout << "XS file: " << parser.get_positional(1) << std::endl;
  }

  // Pass by reference to EigenSolver
  EigenSolver eigen_solver(solver_args, sweeper);

  // Use the eigen_solver
  eigen_solver.solve();
  std::cout << "Final keff: " << eigen_solver.keff() << std::endl;

  Kokkos::finalize();
  return 0;
}
