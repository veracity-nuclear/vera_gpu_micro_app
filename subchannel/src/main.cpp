#include <Kokkos_Core.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include "argument_parser.hpp"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // Create argument parser with pre-configured arguments
    ArgumentParser parser = ArgumentParser::vera_gpu_subchannel_parser(argv[0]);

    // Parse arguments
    if (!parser.parse(argc, argv)) {
        Kokkos::finalize();
        return 1;
    }

    // Get the verbosity and print execution spaces if needed
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
    }

    // Print all the argument options
    if (verbose) {
        // Print positional arguments
        std::cout << "Input file: " << parser.get_positional(0) << std::endl;

        // Print optional arguments
        std::cout << "Verbose: " << (parser.get_flag("verbose") ? "true" : "false") << std::endl;
        std::cout << "Threads: " << parser.get_option("threads") << std::endl;
        std::cout << "Device: " << parser.get_option("device") << std::endl;
        std::cout << "Max Outer Iterations: " << parser.get_option("max_outer_iter") << std::endl;
        std::cout << "Max Inner Iterations: " << parser.get_option("max_inner_iter") << std::endl;
    }
  }
  Kokkos::finalize();
  return 0;
}
