#pragma once

#include <vector>
#include "highfive/highfive.hpp"

class AngFluxBCFace
{
    public:
        std::vector<std::vector<std::vector<double>>> _angflux;
        AngFluxBCFace() = default;
        AngFluxBCFace(int nbc, int npol, int ng) {
            _angflux.resize(nbc);
            for (size_t i = 0; i < nbc; i++) {
                _angflux[i].resize(npol);
                for (size_t j = 0; j < npol; j++) {
                    _angflux[i][j].resize(ng, 0.0);
                }
            }
        };
};
class AngFluxBCAngle
{
    public:
         std::vector<AngFluxBCFace> _faces;
         AngFluxBCAngle() = default;
         AngFluxBCAngle(int nfaces) {
            _faces.resize(nfaces);
        };
};

class LongRay
{
    public:
    std::vector<int> _fsrs;
    std::vector<double> _segments;
    std::vector<std::pair<double, double>> _starting_points;
    std::vector<int> _bc_face;
    std::vector<int> _bc_index;
    double _radians;
    int _angle_index;

    LongRay(const HighFive::Group& group, int angle_index, double radians)
    : _radians(radians),
      _angle_index(angle_index),
      _fsrs(group.getDataSet("FSRs").read<std::vector<int>>()),
      _segments(group.getDataSet("Segments").read<std::vector<double>>()),
      _bc_face(group.getDataSet("BC_face").read<std::vector<int>>()),
      _bc_index(group.getDataSet("BC_index").read<std::vector<int>>())
      {
        auto points_data = group.getDataSet("Starting_Point").read<std::vector<double>>();
        _starting_points.reserve(points_data.size() / 3);
        for (int i = 0; i < points_data.size(); i += 3) {
            _starting_points.push_back(std::make_pair(points_data[i], points_data[i + 1]));
        };
        _bc_face[0] -= 1;
        _bc_face[1] -= 1;
        _bc_index[0] -= 1;
        _bc_index[1] -= 1;
    };
    int angle() const { return _angle_index; };
};
