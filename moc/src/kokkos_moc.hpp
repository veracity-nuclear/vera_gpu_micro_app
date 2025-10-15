#pragma once
#include <string>
#include <vector>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include "c5g7_library.hpp"
#include "base_moc.hpp"
#include "argument_parser.hpp"
#include "kokkos_long_ray.hpp"

static constexpr double fourpi = 4.0 * M_PI;

template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace, typename RealType = float>
class KokkosMOC : public BaseMOC {
    using layout = typename ExecutionSpace::array_layout;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using HViewInt1D = Kokkos::View<int*, layout, Kokkos::HostSpace>;
    using HViewDouble2D = Kokkos::View<double**, layout, Kokkos::HostSpace>;
    using HViewReal1D = Kokkos::View<RealType*, layout, Kokkos::HostSpace>;
    using HViewReal2D = Kokkos::View<RealType**, layout, Kokkos::HostSpace>;
    using HViewReal3D = Kokkos::View<RealType***, layout, Kokkos::HostSpace>;
    using DViewInt1D = Kokkos::View<int*, layout, MemorySpace>;
    using DViewDouble2D = Kokkos::View<double**, layout, MemorySpace>;
    using DViewReal1D = Kokkos::View<RealType*, layout, MemorySpace>;
    using DViewReal2D = Kokkos::View<RealType**, layout, MemorySpace>;
    using DViewReal3D = Kokkos::View<RealType***, layout, MemorySpace>;
    using HViewKokkosRaySegment1D = Kokkos::View<KokkosRaySegment<RealType>*, layout, Kokkos::HostSpace>;
    using DViewKokkosRaySegment1D = Kokkos::View<KokkosRaySegment<RealType>*, layout, MemorySpace>;
    using HViewKokkosLongRay1D = Kokkos::View<KokkosLongRay*, layout, Kokkos::HostSpace>;
    using DViewKokkosLongRay1D = Kokkos::View<KokkosLongRay*, layout, MemorySpace>;

    using simd_real = Kokkos::Experimental::simd<RealType>;
    using DViewSIMD1DReal = Kokkos::View<simd_real*, typename ExecutionSpace::memory_space>;

    // Friend declaration for googletest
    friend class BasicTest_test_kokkos_exp_table_Test;

    public:
        // Constructor
        KokkosMOC(const ArgumentParser& args);

        // Run the MOC sweep
        void sweep() override;

        // Unified implementation of MOC sweep for any execution space
        void _impl_sweep();

        // Get the FSR volumes
        std::vector<double> fsr_vol() const override {
            std::vector<double> result(_nfsr);
            for (int i = 0; i < _nfsr; i++) {
                result[i] = static_cast<double>(_h_fsr_vol(i));
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
        struct RayInfo {
            std::string angle_name;
            std::string ray_name;
            int angle_index;
            int nsegs;
            HighFive::Group ray_group;

            RayInfo(const std::string& ang_name, const std::string& r_name, int ang_idx, int nseg, HighFive::Group group)
                : angle_name(ang_name), ray_name(r_name), angle_index(ang_idx), nsegs(nseg), ray_group(group) {}
        };

        std::vector<RayInfo> _read_ray_infos();
        HViewKokkosLongRay1D _read_rays(std::vector<RayInfo> ray_infos);  // Read rays from the HDF5 file
        HViewKokkosRaySegment1D _read_segments(std::vector<RayInfo> ray_infos);  // Read rays from the HDF5 file
        void _get_xstr(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xstr from XS library
        void _get_xsnf(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xsnf from XS library
        void _get_xsch(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xsch from XS library
        void _get_xssc(const int num_fsr, const std::vector<int>& fsr_mat_id, const c5g7_library& library);  // Read xssc from XS library

        // Input data
        std::string _filename;  // HDF5 file name
        HighFive::File _file; // HDF5 file object
        std::string _device;  // Name of the target Kokkos device
        std::string _ray_sort;  // Ray sorting method

        // Sizes
        int _num_planes;  // Number of planes to simulate
        int _nfsr;  // Number of FSRs
        int _nfsr_per_plane;  // Number of FSRs per plane
        int _npol;  // Number of polar angles
        int _ng;  // Number of energy groups
        int _n_exp_intervals;  // Number of exponential table intervals
        RealType _exp_rdx;  // Exponential table inverse spacing

        // Geometry host data
        RealType _plane_height;  // Height of the plane
        HViewReal1D _h_fsr_vol;
        HViewReal2D _h_xstr;
        HViewReal2D _h_xsnf;
        HViewReal2D _h_xsch;
        HViewReal3D _h_xssc;
        std::vector<RealType> _ray_spacing;
        HViewReal2D _h_exp_table;

        // Geometry device data
        DViewReal1D _d_fsr_vol;
        DViewReal2D _d_xstr;
        DViewReal2D _d_xsnf;
        DViewReal2D _d_xsch;
        DViewReal3D _d_xssc;
        DViewReal2D _d_angle_weights;
        DViewSIMD1DReal _d_rsinpolang;
        DViewReal2D _d_exp_table;

        // Ray data
        int _n_rays;  // Number of rays
        int _n_rays_per_plane;  // Number of rays per plane
        int _max_segments;  // Maximum number of segments in any ray
        DViewKokkosRaySegment1D _d_segments;
        DViewKokkosLongRay1D _d_rays;

        // Solution host data
        HViewDouble2D _h_scalar_flux;
        HViewReal2D _h_source;
        HViewReal3D _h_angflux;
        HViewReal3D _h_old_angflux;

        // Solution device data
        DViewDouble2D _d_scalar_flux;
        DViewReal2D _d_source;
        DViewReal3D _d_angflux;
        DViewReal3D _d_old_angflux;
        DViewReal3D _d_thread_scalar_flux;
};
