#pragma once
#include <vector>
#include <string>
#include "serial_moc.hpp"

double run_eigenvalue_iteration(const std::vector<std::string> args);
class EigenSolver {
    public:
        // Build the solver; first argument is the HDF5 file name, second is the cross-section library name
        EigenSolver(const std::vector<std::string>& args);
        // Run the eigenvalue iteration
        void solve();
        // Get the keff
        double keff();
    private:
        SerialMOC _sweeper;
        std::vector<double> _fsr_vol;
        const std::vector<std::vector<double>> &_scalar_flux;
        std::vector<std::vector<double>> _old_scalar_flux;
        std::vector<double> _fissrc;
        std::vector<double> _old_fissrc;
        double _keff;
        double _old_keff;
        const double _relaxation = 1.0; // Relaxation factor for flux updates
        const int _max_iters = 10000;
        const double _kconv = 1.0e-8;
        const double _fconv = 1.0e-8;
        const int _debug_angle = 0;
};