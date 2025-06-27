#pragma once
#include <string>
#include <vector>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include "c5g7_library.hpp"
#include "base_moc.hpp"
#include "argument_parser.hpp"

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class KokkosMOC : public BaseMOC {
    using layout = typename ExecutionSpace::array_layout;
    using MemorySpace = typename ExecutionSpace::memory_space;

    public:
        // Constructor
        KokkosMOC(const ArgumentParser& args);

        // Run the MOC sweep
        void sweep() override;

        // Unified implementation of MOC sweep for any execution space
        void _impl_sweep();
        
        // Configure team policy based on execution space
        Kokkos::TeamPolicy<ExecutionSpace> _configure_team_policy(int n_rays, int npol, int ng);

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
        void _convert_rays();  // Convert rays to flattened format
        void _get_xstr(int starting_xsr);  // Read xstr from XS library

        // Input data
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        const c5g7_library _library;  // Cross-section library object
        std::string _device;  // Name of the target Kokkos device

        // Sizes
        int _nfsr;  // Number of FSRs
        int _npol;  // Number of polar angles
        int _ng;  // Number of energy groups
        int _n_exp_intervals;  // Number of exponential table intervals
        double _exp_rdx;  // Exponential table inverse spacing

        // Geometry host data
        double _plane_height;  // Height of the plane
        Kokkos::View<double*, layout, Kokkos::HostSpace> _h_fsr_vol;
        std::vector<int> _fsr_mat_id;  // FSR material IDs
        Kokkos::View<double**, layout, Kokkos::HostSpace> _h_xstr;
        std::vector<double> _ray_spacing;
        Kokkos::View<double**, layout, Kokkos::HostSpace> _h_angle_weights;
        Kokkos::View<double*, layout, Kokkos::HostSpace> _h_rsinpolang;
        Kokkos::View<double**, layout, Kokkos::HostSpace> _h_exp_table;

        // Geometry device data
        Kokkos::View<double*, layout, MemorySpace> _d_fsr_vol;
        Kokkos::View<double**, layout, MemorySpace> _d_xstr;
        Kokkos::View<double**, layout, MemorySpace> _d_angle_weights;
        Kokkos::View<double*, layout, MemorySpace> _d_rsinpolang;
        Kokkos::View<double**, layout, MemorySpace> _d_exp_table;

        // Ray host data
        int _n_rays;  // Number of rays
        int _max_segments;  // Maximum number of segments in any ray
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_nsegs;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_bc_face_start;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_bc_face_end;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_bc_index_frwd_start;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_bc_index_frwd_end;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_bc_index_bkwd_start;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_bc_index_bkwd_end;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_angle_index;
        Kokkos::View<int*, layout, Kokkos::HostSpace> _h_ray_fsrs;
        Kokkos::View<double*, layout, Kokkos::HostSpace> _h_ray_segments;

        // Ray device data
        Kokkos::View<int*, layout, MemorySpace> _d_ray_nsegs;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_bc_face_start;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_bc_face_end;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_bc_index_frwd_start;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_bc_index_frwd_end;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_bc_index_bkwd_start;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_bc_index_bkwd_end;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_angle_index;
        Kokkos::View<int*, layout, MemorySpace> _d_ray_fsrs;
        Kokkos::View<double*, layout, MemorySpace> _d_ray_segments;

        // Solution host data
        Kokkos::View<double**, layout, Kokkos::HostSpace> _h_scalar_flux;
        Kokkos::View<double**, layout, Kokkos::HostSpace> _h_source;
        Kokkos::View<double***, layout, Kokkos::HostSpace> _h_angflux;
        Kokkos::View<double***, layout, Kokkos::HostSpace> _h_old_angflux;

        // Solution device data
        Kokkos::View<double**, layout, MemorySpace> _d_scalar_flux;
        Kokkos::View<double**, layout, MemorySpace> _d_source;
        Kokkos::View<double***, layout, MemorySpace> _d_angflux;
        Kokkos::View<double***, layout, MemorySpace> _d_old_angflux;
};
