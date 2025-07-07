#pragma once
#include <string>
#include <vector>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include "c5g7_library.hpp"
#include "base_moc.hpp"
#include "argument_parser.hpp"

static constexpr double fourpi = 4.0 * M_PI;

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class KokkosMOC : public BaseMOC {
    using layout = typename ExecutionSpace::array_layout;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using HViewInt1D = Kokkos::View<int*, layout, Kokkos::HostSpace>;
    using HViewDouble1D = Kokkos::View<double*, layout, Kokkos::HostSpace>;
    using HViewDouble2D = Kokkos::View<double**, layout, Kokkos::HostSpace>;
    using HViewDouble3D = Kokkos::View<double***, layout, Kokkos::HostSpace>;
    using DViewInt1D = Kokkos::View<int*, layout, MemorySpace>;
    using DViewDouble1D = Kokkos::View<double*, layout, MemorySpace>;
    using DViewDouble2D = Kokkos::View<double**, layout, MemorySpace>;
    using DViewDouble3D = Kokkos::View<double***, layout, MemorySpace>;

    // Friend declaration for googletest
    friend class BasicTest_test_kokkos_exp_table_Test;

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
        void _get_xstr(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xstr from XS library
        void _get_xsnf(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xsnf from XS library
        void _get_xsch(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xsch from XS library
        void _get_xssc(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xssc from XS library

        // Input data
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        std::string _device;  // Name of the target Kokkos device

        // Sizes
        int _nfsr;  // Number of FSRs
        int _npol;  // Number of polar angles
        int _ng;  // Number of energy groups
        int _n_exp_intervals;  // Number of exponential table intervals
        double _exp_rdx;  // Exponential table inverse spacing

        // Geometry host data
        double _plane_height;  // Height of the plane
        HViewDouble1D _h_fsr_vol;
        HViewDouble2D _h_xstr;
        HViewDouble2D _h_xsnf;
        HViewDouble2D _h_xsch;
        HViewDouble3D _h_xssc;
        std::vector<double> _ray_spacing;
        HViewDouble2D _h_angle_weights;
        HViewDouble1D _h_rsinpolang;
        HViewDouble2D _h_exp_table;

        // Geometry device data
        DViewDouble1D _d_fsr_vol;
        DViewDouble2D _d_xstr;
        DViewDouble2D _d_xsnf;
        DViewDouble2D _d_xsch;
        DViewDouble3D _d_xssc;
        DViewDouble2D _d_angle_weights;
        DViewDouble1D _d_rsinpolang;
        DViewDouble2D _d_exp_table;

        // Ray host data
        int _n_rays;  // Number of rays
        int _max_segments;  // Maximum number of segments in any ray
        HViewInt1D _h_ray_nsegs;
        HViewInt1D _h_ray_bc_face_start;
        HViewInt1D _h_ray_bc_face_end;
        HViewInt1D _h_ray_bc_index_frwd_start;
        HViewInt1D _h_ray_bc_index_frwd_end;
        HViewInt1D _h_ray_bc_index_bkwd_start;
        HViewInt1D _h_ray_bc_index_bkwd_end;
        HViewInt1D _h_ray_angle_index;
        HViewInt1D _h_ray_fsrs;
        HViewDouble1D _h_ray_segments;

        // Ray device data
        DViewInt1D _d_ray_nsegs;
        DViewInt1D _d_ray_bc_face_start;
        DViewInt1D _d_ray_bc_face_end;
        DViewInt1D _d_ray_bc_index_frwd_start;
        DViewInt1D _d_ray_bc_index_frwd_end;
        DViewInt1D _d_ray_bc_index_bkwd_start;
        DViewInt1D _d_ray_bc_index_bkwd_end;
        DViewInt1D _d_ray_angle_index;
        DViewInt1D _d_ray_fsrs;
        DViewDouble1D _d_ray_segments;

        // Solution host data
        HViewDouble2D _h_scalar_flux;
        HViewDouble2D _h_source;
        HViewDouble3D _h_angflux;
        HViewDouble3D _h_old_angflux;

        // Solution device data
        DViewDouble2D _d_scalar_flux;
        DViewDouble2D _d_source;
        DViewDouble3D _d_angflux;
        DViewDouble3D _d_old_angflux;
        DViewDouble3D _d_thread_scalar_flux;
};
