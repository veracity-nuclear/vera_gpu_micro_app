#include "eigen_solver.hpp"
#include <memory>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include "argument_parser.hpp"
#include "base_moc.hpp"

EigenSolver::EigenSolver(const ArgumentParser& args, std::shared_ptr<BaseMOC> sweeper)
  : _max_iters(std::stoi(args.get_option("max_iter"))),
    _kconv(std::stod(args.get_option("k_conv_crit"))),
    _fconv(std::stod(args.get_option("f_conv_crit"))),
    _sweeper(sweeper),
    _fsr_vol(sweeper->fsr_vol())
  {
  _scalar_flux = _sweeper->scalar_flux();
  _old_scalar_flux = _scalar_flux;

  _keff = 1.0;
  _old_keff = 1.0;
  _fissrc = _sweeper->fission_source(_keff);
  _old_fissrc = _fissrc;
}

void EigenSolver::solve() {

  // Source iteration loop
  std::cout << "Iteration         keff       knorm      fnorm" << std::endl;
  for (int iteration = 0; iteration < _max_iters; iteration++) {
      // Build source and zero the fluxes
      _sweeper->update_source(_fissrc);

      // Execute the MOC sweep
      _sweeper->sweep();

      // Update fission source and keff
      _scalar_flux = _sweeper->scalar_flux();
      _fissrc = _sweeper->fission_source(_keff);
      double numerator = 0.0;
      double denominator = 0.0;
      for (size_t i = 0; i < _fissrc.size(); ++i) {
        numerator += _fissrc[i] * _fsr_vol[i];
        denominator += _old_fissrc[i] * _fsr_vol[i];
      }
      _keff = _old_keff * numerator / denominator;

      // Calculate fission source convergence metric
      double fnorm = 0.0;
      double temp;
      for (size_t i = 0; i < _fissrc.size(); ++i) {
        temp = _fissrc[i] - _old_fissrc[i];
        fnorm += temp * temp;
      }
      fnorm = sqrt(fnorm / double(_fissrc.size()));

      // Calculate the keff convergence metric
      double knorm = _keff - _old_keff;

      // Print the iteration results
      std::cout << " " << std::setw(8) << iteration
                << "   " << std::fixed << std::setprecision(8) << _keff
                << "   " << std::scientific << std::setprecision(2) << knorm
                << "   " << fnorm << std::endl;

      // Check for convergence
      if (fabs(knorm) < _kconv && fabs(fnorm) < _fconv) {
          std::cout << "Converged after " << iteration + 1 << " iterations." << std::endl;
          break;
      }

      // Save the old values
      for (size_t i = 0; i < _fissrc.size(); ++i) {
          for (size_t g = 0; g < _scalar_flux[i].size(); ++g) {
              _old_scalar_flux[i][g] = _relaxation * _scalar_flux[i][g] + (1.0 - _relaxation) * _old_scalar_flux[i][g];
          }
      }
      _old_keff = _keff;
      _fissrc = _sweeper->fission_source(_keff);
      _old_fissrc = _fissrc;
  }
}

double EigenSolver::keff() {
    return _keff;
}
