#pragma once

#include <vector>
#include <highfive/highfive.hpp>
#include "base_moc.hpp"

struct KokkosLongRay
{
    public:
        std::vector<int> _fsrs;
        std::vector<double> _segments;
        std::vector<int> _bc_face;
        std::vector<int> _bc_index;
        double _radians;
        int _angle_index;

        // Constructor that initializes the LongRay object from a HighFive::Group
        // for a specific angle index and radians value
        KokkosLongRay() = default;
        KokkosLongRay(const HighFive::Group& group, int angle_index, double radians)
        : _radians(radians),
          _angle_index(angle_index),
          _fsrs(group.getDataSet("FSRs").read<std::vector<int>>()),
          _segments(group.getDataSet("Segments").read<std::vector<double>>()),
          _bc_face(group.getDataSet("BC_face").read<std::vector<int>>()),
          _bc_index(group.getDataSet("BC_index").read<std::vector<int>>())
          {
            _bc_face[RAY_START] -= 1;
            _bc_face[RAY_END] -= 1;
            _bc_index[RAY_START] -= 1;
            _bc_index[RAY_END] -= 1;
        };
        int angle() const { return _angle_index; };
};
