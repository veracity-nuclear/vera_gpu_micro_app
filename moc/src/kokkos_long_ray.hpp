#pragma once

#include <vector>
#include <highfive/highfive.hpp>
#include <Kokkos_Core.hpp>
#include "base_moc.hpp"

// Kokkos-compatible ray segment structure
template <typename RealType = double>
struct KokkosRaySegment {
    int _fsr;
    RealType _length;

    KOKKOS_INLINE_FUNCTION
    KokkosRaySegment() : _fsr(0), _length(0.0) {}

    KOKKOS_INLINE_FUNCTION
    KokkosRaySegment(int fsr, RealType length) : _fsr(fsr), _length(length) {}
};

// Kokkos-compatible LongRay structure
template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace, typename RealType = double>
struct KokkosLongRay
{
    using layout = typename ExecutionSpace::array_layout;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using HViewInt1D = Kokkos::View<int*, layout, Kokkos::HostSpace>;
    using HViewSegment1D = Kokkos::View<KokkosRaySegment<RealType>*, layout, Kokkos::HostSpace>;
    using DViewInt1D = Kokkos::View<int*, layout, MemorySpace>;
    using DViewSegment1D = Kokkos::View<KokkosRaySegment<RealType>*, layout, MemorySpace>;

    public:
        // Host views
        HViewSegment1D _h_segments;
        HViewInt1D _h_bc_face;     // size 2: [start, end]
        HViewInt1D _h_bc_index;    // size 2: [start, end]

        // Processed boundary condition indices for angular flux mapping
        HViewInt1D _h_angflux_bc_frwd_start;
        HViewInt1D _h_angflux_bc_frwd_end;
        HViewInt1D _h_angflux_bc_bkwd_start;
        HViewInt1D _h_angflux_bc_bkwd_end;

        // Device views
        DViewSegment1D _d_segments;
        DViewInt1D _d_bc_face;
        DViewInt1D _d_bc_index;

        // Device views for processed BC indices
        DViewInt1D _d_angflux_bc_frwd_start;
        DViewInt1D _d_angflux_bc_frwd_end;
        DViewInt1D _d_angflux_bc_bkwd_start;
        DViewInt1D _d_angflux_bc_bkwd_end;

        double _radians;
        int _angle_index;
        int _nsegs;

        // Default constructor
        KokkosLongRay() = default;

        // Constructor that initializes the KokkosLongRay object from a HighFive::Group
        // for a specific angle index and radians value
        KokkosLongRay(const HighFive::Group& group, int angle_index, double radians)
        : _radians(radians), _angle_index(angle_index)
        {
            // Read data from HDF5
            auto fsrs = group.getDataSet("FSRs").read<std::vector<int>>();
            auto segments = group.getDataSet("Segments").read<std::vector<double>>();
            auto bc_face = group.getDataSet("BC_face").read<std::vector<int>>();
            auto bc_index = group.getDataSet("BC_index").read<std::vector<int>>();

            _nsegs = fsrs.size();

            // Allocate host views
            _h_segments = HViewSegment1D("segments", _nsegs);
            _h_bc_face = HViewInt1D("bc_face", 2);
            _h_bc_index = HViewInt1D("bc_index", 2);

            // Allocate BC index arrays (will be populated later)
            _h_angflux_bc_frwd_start = HViewInt1D("angflux_bc_frwd_start", 1);
            _h_angflux_bc_frwd_end = HViewInt1D("angflux_bc_frwd_end", 1);
            _h_angflux_bc_bkwd_start = HViewInt1D("angflux_bc_bkwd_start", 1);
            _h_angflux_bc_bkwd_end = HViewInt1D("angflux_bc_bkwd_end", 1);

            // Copy data to host views
            for (int i = 0; i < _nsegs; i++) {
                _h_segments(i) = KokkosRaySegment<RealType>(fsrs[i], static_cast<RealType>(segments[i]));
            }

            // Adjust BC indices (convert from 1-based to 0-based)
            _h_bc_face(RAY_START) = bc_face[RAY_START] - 1;
            _h_bc_face(RAY_END) = bc_face[RAY_END] - 1;
            _h_bc_index(RAY_START) = bc_index[RAY_START] - 1;
            _h_bc_index(RAY_END) = bc_index[RAY_END] - 1;

            // Create device mirrors and copy data
            _d_segments = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_segments);
            _d_bc_face = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_bc_face);
            _d_bc_index = Kokkos::create_mirror_view_and_copy(MemorySpace(), _h_bc_index);

            // Create device mirrors for BC indices (will be populated later)
            _d_angflux_bc_frwd_start = Kokkos::create_mirror(MemorySpace(), _h_angflux_bc_frwd_start);
            _d_angflux_bc_frwd_end = Kokkos::create_mirror(MemorySpace(), _h_angflux_bc_frwd_end);
            _d_angflux_bc_bkwd_start = Kokkos::create_mirror(MemorySpace(), _h_angflux_bc_bkwd_start);
            _d_angflux_bc_bkwd_end = Kokkos::create_mirror(MemorySpace(), _h_angflux_bc_bkwd_end);
        }

        // Method to set the processed boundary condition indices (called after angular flux setup)
        void set_angflux_bc_indices(int frwd_start, int frwd_end, int bkwd_start, int bkwd_end) {
            _h_angflux_bc_frwd_start(0) = frwd_start;
            _h_angflux_bc_frwd_end(0) = frwd_end;
            _h_angflux_bc_bkwd_start(0) = bkwd_start;
            _h_angflux_bc_bkwd_end(0) = bkwd_end;

            // Copy to device
            Kokkos::deep_copy(_d_angflux_bc_frwd_start, _h_angflux_bc_frwd_start);
            Kokkos::deep_copy(_d_angflux_bc_frwd_end, _h_angflux_bc_frwd_end);
            Kokkos::deep_copy(_d_angflux_bc_bkwd_start, _h_angflux_bc_bkwd_start);
            Kokkos::deep_copy(_d_angflux_bc_bkwd_end, _h_angflux_bc_bkwd_end);
        }

        int angle() const { return _angle_index; }
        int nsegs() const { return _nsegs; }

        // Get FSR for a given segment (device version)
        KOKKOS_INLINE_FUNCTION
        int d_fsr(int iseg) const {
            return _d_segments(iseg)._fsr;
        }

        // Get segment length for a given segment (device version)
        KOKKOS_INLINE_FUNCTION
        RealType d_segment(int iseg) const {
            return _d_segments(iseg)._length;
        }

        // Get BC face (device version)
        KOKKOS_INLINE_FUNCTION
        int d_bc_face(int end) const {
            return _d_bc_face(end);
        }

        // Get BC index (device version)
        KOKKOS_INLINE_FUNCTION
        int d_bc_index(int end) const {
            return _d_bc_index(end);
        }

        // Get processed angular flux BC indices (device version)
        KOKKOS_INLINE_FUNCTION
        int d_angflux_bc_frwd_start() const {
            return _d_angflux_bc_frwd_start(0);
        }

        KOKKOS_INLINE_FUNCTION
        int d_angflux_bc_frwd_end() const {
            return _d_angflux_bc_frwd_end(0);
        }

        KOKKOS_INLINE_FUNCTION
        int d_angflux_bc_bkwd_start() const {
            return _d_angflux_bc_bkwd_start(0);
        }

        KOKKOS_INLINE_FUNCTION
        int d_angflux_bc_bkwd_end() const {
            return _d_angflux_bc_bkwd_end(0);
        }

        // Get FSR for a given segment (host version)
        KOKKOS_INLINE_FUNCTION
        int fsr(int iseg) const {
            return _h_segments(iseg)._fsr;
        }

        // Get segment length for a given segment (host version)
        KOKKOS_INLINE_FUNCTION
        RealType segment(int iseg) const {
            return _h_segments(iseg)._length;
        }

        // Get BC face (host version)
        KOKKOS_INLINE_FUNCTION
        int bc_face(int end) const {
            return _h_bc_face(end);
        }

        // Get BC index (host version)
        KOKKOS_INLINE_FUNCTION
        int bc_index(int end) const {
            return _h_bc_index(end);
        }
};
