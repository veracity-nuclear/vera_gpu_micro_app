#include <Kokkos_Core.hpp>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
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

// Helper function to create SerialMOC with specified precision
std::shared_ptr<BaseMOC> create_serial_moc_with_precision(const ArgumentParser& parser) {
    std::string precision = parser.get_option("precision");
    if (precision == "single") {
        return std::make_shared<SerialMOC<float>>(parser);
    } else if (precision == "double") {
        return std::make_shared<SerialMOC<double>>(parser);
    } else {
        throw std::runtime_error("Unsupported precision: " + precision);
    }
}

int main(int argc, char *argv[]) {
  const int num_runs = 5;
  std::vector<double> wallclock_times;
  double runtime = 0.;
  int next_run = 0;

  // Create argument parser with pre-configured arguments
  // remove custom positional (place 3)
  std::string out_filename = argv[3];
  char* new_args[argc - 1];
  for ( int i = 0; i < argc; i++ ) {
    if ( i < 3 ) {
      new_args[i] = argv[i];
    } else if ( i > 3 ) {
      new_args[i-1] = argv[i];
    }
  }

  ArgumentParser parser = ArgumentParser::vera_gpu_moc_parser(argv[0]);
  if (!parser.parse(argc-1, new_args)) {
    return 1;
  }
  
  bool verbose = parser.get_flag("verbose");

  std::ofstream out_file;
  out_file.open(out_filename, std::ios::app);
  
  {
    Kokkos::initialize(argc, argv);
    {
      
      int concurrency = Kokkos::DefaultExecutionSpace::concurrency();
      std::cout << "\t " << std::to_string(concurrency) << " Kokkos Concurrency" << std::endl;
      
      // Read the sweeper argument
      std::string sweeper_type = parser.get_option("sweeper");
      // std::string sweeper_type = "kokkos"; 
     
      std::string device_type = parser.get_option("device");
 
      // Set up the sweeper
      std::shared_ptr<BaseMOC> sweeper;
      
      // Loop and time the eigen_solver
      if constexpr(true) {    // future profiling condition
        Kokkos::Profiling::pushRegion("Solve Loop");
      }
      for (int r = next_run; r < num_runs; r++){

        if constexpr(true) {    // future profiling condition
          Kokkos::Profiling::pushRegion(("Sweeper and Solver Setup "+std::to_string(r)).c_str());
        }
       
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
          } else if (device_type == "sycl") {
              #ifdef KOKKOS_ENABLE_SYCL
              sweeper = create_kokkos_moc_with_precision<Kokkos::SYCL>(parser);
              #else
              std::cerr << "SYCL execution space not enabled in Kokkos!" << std::endl;
              Kokkos::finalize();
              return 1;
              #endif
          } else {
              throw std::runtime_error("Unsupported Kokkos execution space: " + device_type);
          }
        } else if (sweeper_type == "serial") {
            sweeper = create_serial_moc_with_precision(parser);
        }

        // Pass by reference to EigenSolver
        EigenSolver eigen_solver(parser, sweeper);

        if constexpr(true) {    // future profiling condition
          Kokkos::Profiling::popRegion();
          Kokkos::Profiling::pushRegion(("Solve "+std::to_string(r)).c_str());
        }

        runtime = eigen_solver.time_solve();

        if constexpr(true) {    // future profiling condition
          Kokkos::Profiling::popRegion();
        }

        if ( verbose ) {
          std::cout << "wallclock time " << r << ": " << runtime << std::endl;
        }
        // run_file << runtime << "\t";
        wallclock_times.push_back(runtime);
      }
      if constexpr(true) {    // future profiling condition
        Kokkos::Profiling::popRegion();
      }
      // sample mean time
      double wallclock_mean = std::accumulate(wallclock_times.begin(),wallclock_times.end(),0.0);
      wallclock_mean /= (double)num_runs;
      // approximate unbiased sample standard deviation
      std::vector<double> wallclock_devsq(num_runs);
      std::transform(wallclock_times.begin(), wallclock_times.end(), wallclock_devsq.begin(), [wallclock_mean](double x) { return pow(x - wallclock_mean,2); });
      double wallclock_std = std::accumulate(wallclock_devsq.begin(),wallclock_devsq.end(),0.0);
      wallclock_std = sqrt(
        1. / ((double)num_runs - 1.5 + (1. / (double)(8*(num_runs - 1)))) * wallclock_std
      );
      double wallclock_ci = 2. * wallclock_std;

      // sample median time
      double wallclock_median;
      int n = wallclock_times.size() / 2;
      std::sort(wallclock_times.begin(), wallclock_times.end());
      if (n % 2 == 0) {
          // Even number of elements: average of the two middle elements
          wallclock_median = (wallclock_times[n / 2 - 1] + wallclock_times[n / 2]) / 2.0;
      } else {
          // Odd number of elements: middle element
          wallclock_median = wallclock_times[n / 2];
      }

      std::cout << "Median wallclock time: " << wallclock_median << std::endl;

      std::cout << "Mean wallclock time: " << wallclock_mean;
      std::cout << " +- " << wallclock_ci << std::endl;
      
      out_file << concurrency << "\t" << wallclock_median << "\t" <<  wallclock_mean << "\t" << wallclock_ci << std::endl;
      out_file.close();
      }
      Kokkos::finalize();
    }
    return 0;
}
