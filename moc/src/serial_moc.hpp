#pragma once
#include <string>
#include <vector>
#include <highfive/highfive.hpp>
#include "exp_table.hpp"
#include "c5g7_library.hpp"
#include "long_ray.hpp"
#include "base_moc.hpp"
#include "argument_parser.hpp"

class SerialMOC : public BaseMOC {
    public:
        // Constructor
        SerialMOC(const ArgumentParser& args);

        // Run the MOC sweep
        void sweep() override;

        // Get the FSR volumes
        std::vector<double> fsr_vol() const override { return _fsr_vol; }

        // Get the scalar flux
        std::vector<std::vector<double>> scalar_flux() const override { return _scalar_flux; }

        // Calculate the fission source
        std::vector<double> fission_source(const double keff) const override;

        // Set the total source
        void update_source(const std::vector<double>& fissrc) override;

    private:
        void _read_rays();  // Read rays from the HDF5 file
        void _get_xstr(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xstr from XS library
        void _get_xsnf(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xsnf from XS library
        void _get_xsch(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xsch from XS library
        void _get_xssc(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xssc from XS library

        size_t _max_segments;  // Maximum number of segments in any ray
        int _nfsr;  // Number of FSRs
        int _npol;  // Number of polar angles
        int _ng;  // Number of energy groups
        double _plane_height;  // Height of the plane
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        const ExpTable _exp_table;  // Exponential table for calculations
        std::vector<std::vector<double>> _expoa;  // Inline exponential table slope coefficients
        std::vector<std::vector<double>> _expob;  // Inline exponential table intercept coefficients
        std::vector<double> _fsr_vol;  // FSR volumes
        std::vector<std::vector<double>> _xstr;  // Transport cross-sections for each FSR
        std::vector<std::vector<double>> _xsnf;  // Nu-fission cross-sections for each FSR
        std::vector<std::vector<double>> _xsch;  // Chi for each FSR
        std::vector<std::vector<std::vector<double>>> _xssc;  // Scattering cross-sections for each FSR
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
