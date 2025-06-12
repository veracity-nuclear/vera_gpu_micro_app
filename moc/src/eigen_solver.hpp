#pragma once
#include <vector>
#include <string>
#include "base_moc.hpp"

class EigenSolver {
    public:
        // Build the solver; first argument is the HDF5 file name, second is the cross-section library name
        EigenSolver(const std::vector<std::string> args, BaseMOC* sweeper);
        // Run the eigenvalue iteration
        void solve();
        // Get the keff
        double keff();
    private:
        const int _max_iters = 10000; // Maximum number of iterations
        const double _relaxation = 1.0; // Relaxation factor for flux updates
        const double _kconv = 1.0e-8; // Convergence criterion for keff
        const double _fconv = 1.0e-8;  // Convergence criterion for fission source
        double _keff; // Current eigenvalue estimate
        double _old_keff; // Previous eigenvalue estimate
        const std::vector<double> _fsr_vol; // FSR volumes
        std::vector<std::vector<double>> _scalar_flux; // Scalar flux from the sweeper
        std::vector<std::vector<double>> _old_scalar_flux; // Previous scalar flux for convergence checks
        std::vector<double> _fissrc; // Fission source vector
        std::vector<double> _old_fissrc; // Previous fission source for convergence checks
	BaseMOC* _sweeper; // MOC sweeper object
};
