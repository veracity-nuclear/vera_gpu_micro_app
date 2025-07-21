#include <Kokkos_Core.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include "argument_parser.hpp"
#include "base_moc.hpp"
#include "serial_moc.hpp"
#include "kokkos_moc.hpp"
#include "eigen_solver.hpp"

// Helper template function to create KokkosMOC with specified precision
template<typename ExecutionSpace>
std::shared_ptr<BaseMOC> create_kokkos_moc_with_precision(const ArgumentParser& parser) {
    std::string precision = parser.get_option("precision");
    if (precision == "single") {
        return std::make_shared<KokkosMOC<ExecutionSpace, float>>(parser);
    } else if (precision == "double") {
        return std::make_shared<KokkosMOC<ExecutionSpace, double>>(parser);
    } else {
        throw std::runtime_error("Unsupported precision: " + precision);
    }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // Create argument parser with pre-configured arguments
    ArgumentParser parser = ArgumentParser::vera_gpu_moc_parser(argv[0]);

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
        std::cout << "Precision: " << parser.get_option("precision") << std::endl;
    } else {
        if (sweeper_type == "kokkos") {
            std::string device = parser.get_option("device");
            std::cout << "Selected kokkos backend: " << device << std::endl;
        }
    }

    // Set up the sweeper
    std::shared_ptr<BaseMOC> sweeper;
    std::string device_type = parser.get_option("device");
    std::string precision = parser.get_option("precision");
    std::cout << "Creating sweeper type " << sweeper_type << " targeting device " << device_type << " and " << precision << " precision" << std::endl;

    if (sweeper_type == "kokkos") {
        if (device_type == "serial") {
            #ifdef KOKKOS_ENABLE_SERIAL
            sweeper = create_kokkos_moc_with_precision<Kokkos::Serial>(parser);
            #else
            std::cerr << "Serial execution space not enabled in Kokkos!" << std::endl;
            Kokkos::finalize();
            return 1;
            #endif
        } else if (device_type == "openmp") {
            #ifdef KOKKOS_ENABLE_OPENMP
            sweeper = create_kokkos_moc_with_precision<Kokkos::OpenMP>(parser);
            #else
            std::cerr << "OpenMP execution space not enabled in Kokkos!" << std::endl;
            Kokkos::finalize();
            return 1;
            #endif
        } else if (device_type == "cuda") {
            #ifdef KOKKOS_ENABLE_CUDA
            sweeper = create_kokkos_moc_with_precision<Kokkos::Cuda>(parser);
            #else
            std::cerr << "CUDA execution space not enabled in Kokkos!" << std::endl;
            Kokkos::finalize();
            return 1;
            #endif
        } else {
            throw std::runtime_error("Unsupported Kokkos execution space: " + device_type);
        }
    } else if (sweeper_type == "serial") {
        sweeper = std::make_shared<SerialMOC>(parser);
    }

    // Pass by reference to EigenSolver
    EigenSolver eigen_solver(parser, sweeper);

    // Use the eigen_solver
    eigen_solver.solve();
    std::cout << "Final keff: " << eigen_solver.keff() << std::endl;
  }
  Kokkos::finalize();
  return 0;
}
