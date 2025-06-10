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
        void _impl_sweep_openmp();  // Implementation of the MOC sweep using OpenMP
        void _impl_sweep_serial();  // Implementation of the MOC sweep using serial

        // Get the FSR volumes
        std::vector<double> fsr_vol() const override {
            std::vector<double> result(_nfsr);
            for (int i = 0; i < _nfsr; i++) {
                result[i] = _h_fsr_vol(i);
            }
            return result;
	}

        // Get the scalar flux
        std::vector<std::vector<double>> scalar_flux() const override {
            std::vector<std::vector<double>> result(_nfsr, std::vector<double>(_ng));
            for (int i = 0; i < _nfsr; ++i) {
                for (int g = 0; g < _ng; ++g) {
                    result[i][g] = _h_scalar_flux(i, g);
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
	void _get_xstr(int starting_xsr);  // Read xstr from XS library

	// Input data
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        const c5g7_library _library;  // Cross-section library object
        std::string _device;  // Name of the target Kokkos device

	// Sizes
        size_t _max_segments;  // Maximum number of segments in any ray
        int _nfsr;  // Number of FSRs
        int _npol;  // Number of polar angles
        int _ng;  // Number of energy groups

	// Geometry and ray host data
        double _plane_height;  // Height of the plane
	Kokkos::View<double*, Kokkos::HostSpace> _h_fsr_vol;  // Host copy of FSR volumes
        std::vector<int> _fsr_mat_id;  // FSR material IDs
	Kokkos::View<double**, Kokkos::HostSpace> _h_xstr;  // Host copy of cross-sections for each FSR
	Kokkos::View<KokkosLongRay*, Kokkos::HostSpace> _h_rays;  // Host copy of long rays for MOC
        std::vector<double> _ray_spacing;
        std::vector<std::vector<double>> _angle_weights;  // Weights for each angle
        Kokkos::View<double*, Kokkos::HostSpace> _h_rsinpolang;  // Host copy of precomputed sin(polar angle) values
        Kokkos::View<double**, Kokkos::HostSpace> _h_exparg;  // Exponential arguments for each segment and group

	// Geometry and ray device data
        Kokkos::View<double*> _d_fsr_vol;  // FSR volumes
        Kokkos::View<double**> _d_xstr;  // Cross-sections for each FSR
        Kokkos::View<KokkosLongRay*> _d_rays;  // Long rays for MOC
        Kokkos::View<double*> _d_rsinpolang;  // Precomputed sin(polar angle) values

	// Solution host data
        Kokkos::View<double***, Kokkos::HostSpace> _h_segflux;  // Segment flux array
	Kokkos::View<double**, Kokkos::HostSpace> _h_scalar_flux;  // Host copy of scalar flux array
	Kokkos::View<double**, Kokkos::HostSpace> _h_source;  // Host copy of multigroup total source term for each FSR
        Kokkos::View<double***, Kokkos::HostSpace> _h_angflux;  // Host copy of angular flux for each angle
        Kokkos::View<double***, Kokkos::HostSpace> _h_old_angflux;  // Host copy of old angular flux for each angle

	// Solution device data
        Kokkos::View<double**> _d_scalar_flux;  // Scalar flux array
        Kokkos::View<double**> _d_source;  // Multrigroup total source term for each FSR
};
