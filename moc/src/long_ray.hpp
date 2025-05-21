#pragma once

#include <vector>
#include "highfive/highfive.hpp"

class LongRay
{
    protected:
        std::vector<int> _fsrs;
        std::vector<double> _segments;
        std::vector<std::pair<double, double>> _starting_points;
        double _radians;
    public:
        LongRay(const HighFive::Group& group, double radians) : _radians(radians) {
            _fsrs = group.getDataSet("FSRs").read<std::vector<int>>();
            _segments = group.getDataSet("Segments").read<std::vector<double>>();
            auto points_data = group.getDataSet("Starting_Point").read<std::vector<double>>();
            _starting_points.reserve(points_data.size() / 3);
            for (int i = 0; i < points_data.size(); i += 3) {
                _starting_points.push_back(std::make_pair(points_data[i], points_data[i + 1]));
            };
        };
};
