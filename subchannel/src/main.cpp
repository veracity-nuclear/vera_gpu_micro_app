#include <Kokkos_Core.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include "argument_parser.hpp"
#include "solver.hpp"

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

		{
			std::cout << "Kokkos execution spaces enabled:\n";
			#ifdef KOKKOS_ENABLE_SERIAL
				std::cout << "  - SERIAL\n";
			#endif
			#ifdef KOKKOS_ENABLE_OPENMP
				std::cout << "  - OPENMP\n";
			#endif
			#ifdef KOKKOS_ENABLE_CUDA
				std::cout << "  - CUDA\n";
			#endif
		}

		std::string device = parser.get_option("device");

		if (device == "serial") {
			#ifndef KOKKOS_ENABLE_SERIAL
				std::cerr << "Error: SERIAL execution space is not enabled in Kokkos build." << std::endl;
				Kokkos::finalize();
				return 1;
			#endif
			Solver<Kokkos::Serial> solver(parser);
			// solver.solve();
		} else if (device == "openmp") {
			#ifndef KOKKOS_ENABLE_OPENMP
				std::cerr << "Error: OPENMP execution space is not enabled in Kokkos build." << std::endl;
				Kokkos::finalize();
				return 1;
			#endif
			Solver<Kokkos::OpenMP> solver(parser);
			// solver.solve();
		} else if (device == "cuda") {
			#ifndef KOKKOS_ENABLE_CUDA
				std::cerr << "Error: CUDA execution space is not enabled in Kokkos build." << std::endl;
				Kokkos::finalize();
				return 1;
			#endif
			Solver<Kokkos::Cuda> solver(parser);
			// solver.solve();
		} else {
			std::cerr << "Error: Unsupported device '" << device << "'." << std::endl;
			Kokkos::finalize();
			return 1;
		}
	}
	Kokkos::finalize();
	return 0;
}
