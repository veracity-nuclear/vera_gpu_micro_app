#pragma once

#include <vector>
#include <highfive/highfive.hpp>
#include "base_moc.hpp"

struct KokkosLongRay
{
    public:
        int _nsegs;
        std::vector<int> _fsrs;
        std::vector<double> _segments;
        int _bc_face_start;
        int _bc_face_end;
        int _bc_index_frwd_start;
        int _bc_index_bkwd_start;
        int _bc_index_frwd_end;
        int _bc_index_bkwd_end;
        int _angle_index;

        // Constructor that initializes the LongRay object from a HighFive::Group
        // for a specific angle index and radians value
        KokkosLongRay() = default;
        KokkosLongRay(const HighFive::Group& group, int angle_index)
        : _angle_index(angle_index),
          _fsrs(group.getDataSet("FSRs").read<std::vector<int>>()),
          _segments(group.getDataSet("Segments").read<std::vector<double>>())
          {
            _nsegs = _segments.size();
            auto tmp_bc = group.getDataSet("BC_face").read<std::vector<int>>();
            _bc_face_start = tmp_bc[RAY_START] - 1;
            _bc_face_end = tmp_bc[RAY_END] - 1;
            tmp_bc = group.getDataSet("BC_index").read<std::vector<int>>();
            _bc_index_frwd_start = tmp_bc[RAY_START] - 1;
            _bc_index_frwd_end = tmp_bc[RAY_END] - 1;
            _bc_index_bkwd_start = _bc_index_frwd_end;
            _bc_index_bkwd_end = _bc_index_frwd_start;
        };
        int angle() const { return _angle_index; };
};
