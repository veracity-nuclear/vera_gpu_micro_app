#include "eigen_solver.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include "serial_moc.hpp"

double run_eigenvalue_iteration(const std::vector<std::string> args) {
  SerialMOC sweeper(args[1], args[2]);
  auto scalar_flux = sweeper.scalar_flux();
  auto fsr_vol = sweeper.fsr_vol();
  auto old_scalar_flux = scalar_flux;

  double keff = 1.0;
  double old_keff = 1.0;
  std::vector<double> fissrc = sweeper.fission_source(keff);
  std::vector<double> old_fissrc = fissrc;

  // Initialize iteration controls and print the header
  std::cout << "Iteration         keff       knorm      fnorm" << std::endl;
  double relaxation = 1.0;
  int max_iters = 10000;
  double kconv = 1.0e-8;
  double fconv = 1.0e-8;
  int debug_angle = 0;

  // Source iteration loop
  for (int iteration = 0; iteration < max_iters; iteration++) {
      // Build source and zero the fluxes
      sweeper.update_source(fissrc);

      // Execute the MOC sweep
      sweeper.sweep();

      // Update fission source and keff
      fissrc = sweeper.fission_source(keff);
      double numerator = 0.0;
      double denominator = 0.0;
      for (size_t i = 0; i < fissrc.size(); ++i) {
        numerator += fissrc[i] * fsr_vol[i];
        denominator += old_fissrc[i] * fsr_vol[i];
      }
      keff = old_keff * numerator / denominator;

      // Calculate fission source convergence metric
      double fnorm = 0.0;
      double temp;
      for (size_t i = 0; i < fissrc.size(); ++i) {
        temp = fissrc[i] - old_fissrc[i];
        fnorm += temp * temp;
      }
      fnorm = sqrt(fnorm / double(fissrc.size()));

      // Calculate the keff convergence metric
      double knorm = keff - old_keff;

      // Print the iteration results
      std::cout << " " << std::setw(8) << iteration
                << "   " << std::fixed << std::setprecision(8) << keff
                << "   " << std::scientific << std::setprecision(2) << knorm
                << "   " << fnorm << std::endl;

      // Check for convergence
      if (fabs(knorm) < kconv && fabs(fnorm) < fconv) {
          std::cout << "Converged after " << iteration + 1 << " iterations." << std::endl;
          break;
      }

      // Save the old values
      for (size_t i = 0; i < fissrc.size(); ++i) {
          for (size_t g = 0; g < scalar_flux[i].size(); ++g) {
              old_scalar_flux[i][g] = relaxation * scalar_flux[i][g] + (1.0 - relaxation) * old_scalar_flux[i][g];
          }
      }
      old_keff = keff;
      fissrc = sweeper.fission_source(keff);
      old_fissrc = fissrc;
  }
  return keff;
}