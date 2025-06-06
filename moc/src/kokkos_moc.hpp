#pragma once
#include <string>
#include <vector>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include "c5g7_library.hpp"
#include "kokkos_long_ray.hpp"
#include "base_moc.hpp"
#include "argument_parser.hpp"

class KokkosMOC : public BaseMOC {
    public:
        // Constructor
        KokkosMOC(const ArgumentParser& args);

        // Run the MOC sweep
        void sweep() override;

        // Get the FSR volumes
        const std::vector<double>& fsr_vol() const override { return _fsr_vol; }

        // Get the scalar flux
        std::vector<std::vector<double>> scalar_flux() const override {
            std::vector<std::vector<double>> result(_nfsr, std::vector<double>(_ng));
            for (int i = 0; i < _nfsr; ++i) {
                for (int g = 0; g < _ng; ++g) {
                    result[i][g] = _scalar_flux(i, g);
                }
            }
            return result;
        }

        // Calculate the fission source
        std::vector<double> fission_source(const double keff) const override;

        // Set the total source
        void update_source(const std::vector<double>& fissrc) override;

    private:
        void _read_rays();  // Read rays from the HDF5 file
        void _impl_sweep_openmp();  // Implementation of the MOC sweep using OpenMP
        void _impl_sweep_serial();  // Implementation of the MOC sweep using serial

        size_t _max_segments;  // Maximum number of segments in any ray
        int _nfsr;  // Number of FSRs
        int _npol;  // Number of polar angles
        int _ng;  // Number of energy groups
        double _plane_height;  // Height of the plane
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        const c5g7_library _library;  // Cross-section library object
        std::vector<double> _fsr_vol;  // FSR volumes
        std::vector<int> _fsr_mat_id;  // FSR material IDs
        Kokkos::View<double**> _xstr;  // Cross-sections for each FSR
        Kokkos::View<KokkosLongRay*> _rays;  // Long rays for MOC
        std::vector<double> _ray_spacing;
        std::vector<std::vector<double>> _angle_weights;  // Weights for each angle
        Kokkos::View<double*> _rsinpolang;  // Precomputed sin(polar angle) values
        Kokkos::View<double***> _segflux;  // Segment flux array
        Kokkos::View<double**> _exparg;  // Exponential arguments for each segment and group
        Kokkos::View<double**> _scalar_flux;  // Scalar flux array
        Kokkos::View<double**> _source;  // Multrigroup total source term for each FSR
        std::vector<AngFluxBCAngle> _angflux;  // Angular flux for each angle
        std::vector<AngFluxBCAngle> _old_angflux;  // Angular flux for each angle
        std::string _device;  // Name of the target Kokkos device
};