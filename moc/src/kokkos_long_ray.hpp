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
    public:
        // Ray metadata
        int _nsegs;
        int _bc_face[2];     // size 2: [start, end]
        int _bc_index[2];    // size 2: [start, end]

        // Processed boundary condition indices for angular flux mapping
        int _angflux_bc_frwd_start;
        int _angflux_bc_frwd_end;
        int _angflux_bc_bkwd_start;
        int _angflux_bc_bkwd_end;

        // Angle data
        int _angle_index;

        // Ray segments
        KokkosRaySegment<RealType>* _segments;

        // Default constructor
        KokkosLongRay() = default;

        // Constructor that initializes the KokkosLongRay object from a HighFive::Group
        // for a specific angle index
        KokkosLongRay(const HighFive::Group& group, int angle_index)
        : _angle_index(angle_index)
        {
            // Read data from HDF5
            auto fsrs = group.getDataSet("FSRs").read<std::vector<int>>();
            auto segments = group.getDataSet("Segments").read<std::vector<double>>();
            auto bc_face = group.getDataSet("BC_face").read<std::vector<int>>();
            auto bc_index = group.getDataSet("BC_index").read<std::vector<int>>();

            _nsegs = fsrs.size();

            // Allocate host views
            _segments = new KokkosRaySegment<RealType>[_nsegs];

            // Copy data to host views
            for (int i = 0; i < _nsegs; i++) {
                _segments[i] = KokkosRaySegment<RealType>(fsrs[i], static_cast<RealType>(segments[i]));
            }

            // Adjust BC indices (convert from 1-based to 0-based)
            _bc_face[RAY_START] = bc_face[RAY_START] - 1;
            _bc_face[RAY_END] = bc_face[RAY_END] - 1;
            _bc_index[RAY_START] = bc_index[RAY_START] - 1;
            _bc_index[RAY_END] = bc_index[RAY_END] - 1;

        }

        // Method to set the processed boundary condition indices (called after angular flux setup)
        void set_angflux_bc_indices(int frwd_start, int frwd_end, int bkwd_start, int bkwd_end) {
            _angflux_bc_frwd_start = frwd_start;
            _angflux_bc_frwd_end = frwd_end;
            _angflux_bc_bkwd_start = bkwd_start;
            _angflux_bc_bkwd_end = bkwd_end;
        }

        KOKKOS_INLINE_FUNCTION
        int angle() const {
            return _angle_index;
        }

        KOKKOS_INLINE_FUNCTION
        int nsegs() const {
            return _nsegs;
        }

        // Get FSR for a given segment (device version)
        KOKKOS_INLINE_FUNCTION
        int fsr(int iseg) const {
            return _segments[iseg]._fsr;
        }

        // Get segment length for a given segment (device version)
        KOKKOS_INLINE_FUNCTION
        RealType segment(int iseg) const {
            return _segments[iseg]._length;
        }

        // Get BC face (device version)
        KOKKOS_INLINE_FUNCTION
        int bc_face(int end) const {
            return _bc_face[end];
        }

        // Get BC index (device version)
        KOKKOS_INLINE_FUNCTION
        int bc_index(int end) const {
            return _bc_index[end];
        }

        // Get processed angular flux BC indices (device version)
        KOKKOS_INLINE_FUNCTION
        int angflux_bc_frwd_start() const {
            return _angflux_bc_frwd_start;
        }

        KOKKOS_INLINE_FUNCTION
        int angflux_bc_frwd_end() const {
            return _angflux_bc_frwd_end;
        }

        KOKKOS_INLINE_FUNCTION
        int angflux_bc_bkwd_start() const {
            return _angflux_bc_bkwd_start;
        }

        KOKKOS_INLINE_FUNCTION
        int angflux_bc_bkwd_end() const {
            return _angflux_bc_bkwd_end;
        }
};
