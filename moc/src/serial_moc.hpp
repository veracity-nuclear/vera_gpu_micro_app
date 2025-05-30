#pragma once
#include <string>
#include <vector>
#include <highfive/H5Easy.hpp>
#include <highfive/highfive.hpp>
#include "exp_table.hpp"
#include "c5g7_library.hpp"
#include "long_ray.hpp"

class SerialMOC {
    public:
        // Constructor
        SerialMOC(const std::string& filename, const std::string& libname);
        // Run the MOC sweep
        void sweep();
        // Get the FSR volumes
        const std::vector<double>& fsr_vol() const { return _fsr_vol; }
        // Get the scalar flux
        const std::vector<std::vector<double>>& scalar_flux() const { return _scalar_flux; }
        // Calculate the fission source
        std::vector<double> fission_source(const double keff) const;
        // Set the total source
        void update_source(const std::vector<double>& fissrc);
    private:
        void _read_rays();  // Read rays from the HDF5 file

        size_t _max_segments;  // Maximum number of segments in any ray
        int _nfsr;  // Number of FSRs
        int _npol;  // Number of polar angles
        int _ng;  // Number of energy groups
        double _plane_height;  // Height of the plane
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        const c5g7_library _library;  // Cross-section library object
        const ExpTable _exp_table;  // Exponential table for calculations
        std::vector<double> _fsr_vol;  // FSR volumes
        std::vector<int> _fsr_mat_id;  // FSR material IDs
        std::vector<std::vector<double>> _xstr;  // Cross-sections for each FSR
        std::vector<LongRay> _rays;  // Long rays for MOC
        std::vector<double> _ray_spacing;
        std::vector<std::vector<double>> _angle_weights;  // Weights for each angle
        std::vector<double> _rsinpolang;  // Precomputed sin(polar angle) values for ray tracing
        std::vector<std::vector<std::vector<double>>> _segflux;  // Segment flux array
        std::vector<std::vector<double>> _exparg;  // Exponential arguments for each segment and group
        std::vector<std::vector<double>> _scalar_flux;  // Scalar flux array
        std::vector<std::vector<double>> _source;  // Multrigroup total source term for each FSR
        std::vector<AngFluxBCAngle> _angflux;  // Angular flux for each angle
        std::vector<AngFluxBCAngle> _old_angflux;  // Angular flux for each angle
};