#pragma once
#include <string>
#include <vector>
#include <highfive/highfive.hpp>
#include "exp_table.hpp"
#include "c5g7_library.hpp"
#include "long_ray.hpp"
#include "base_moc.hpp"
#include "argument_parser.hpp"

template <typename RealType>
class SerialMOC : public BaseMOC {
    public:
        // Constructor
        SerialMOC(const ArgumentParser& args);

        // Run the MOC sweep
        void sweep() override;

        // Get the FSR volumes
        std::vector<double> fsr_vol() const override { 
            std::vector<double> result(_fsr_vol.size());
            for (size_t i = 0; i < _fsr_vol.size(); i++) {
                result[i] = static_cast<double>(_fsr_vol[i]);
            }
            return result;
        }

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
        RealType _plane_height;  // Height of the plane
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        const ExpTable _exp_table;  // Exponential table for calculations
        std::vector<RealType> _fsr_vol;  // FSR volumes
        std::vector<std::vector<RealType>> _xstr;  // Transport cross-sections for each FSR
        std::vector<std::vector<RealType>> _xsnf;  // Nu-fission cross-sections for each FSR
        std::vector<std::vector<RealType>> _xsch;  // Chi for each FSR
        std::vector<std::vector<std::vector<RealType>>> _xssc;  // Scattering cross-sections for each FSR
        std::vector<LongRay> _rays;  // Long rays for MOC
        std::vector<RealType> _ray_spacing;
        std::vector<std::vector<RealType>> _angle_weights;  // Weights for each angle
        std::vector<RealType> _rsinpolang;  // Precomputed sin(polar angle) values for ray tracing
        std::vector<std::vector<std::vector<RealType>>> _segflux;  // Segment flux array
        std::vector<std::vector<RealType>> _exparg;  // Exponential arguments for each segment and group
        std::vector<std::vector<double>> _scalar_flux;  // Scalar flux array (stays double)
        std::vector<std::vector<RealType>> _source;  // Multrigroup total source term for each FSR
        std::vector<AngFluxBCAngleT<RealType>> _angflux;  // Angular flux for each angle
        std::vector<AngFluxBCAngleT<RealType>> _old_angflux;  // Angular flux for each angle
};
