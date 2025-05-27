#pragma once

#include <vector>
#include "highfive/highfive.hpp"

const int RAY_START = 0; // Index for the start of the ray
const int RAY_END = 1;   // Index for the end of the ray
const int WEST = 0;    // Index for the west face
const int NORTH = 1;   // Index for the north face
const int EAST = 2;    // Index for the east face
const int SOUTH = 3;   // Index for the south face

// Defines the angular flux boundary condition for a single face and a single angle
class AngFluxBCFace
{
    public:
        // A 3D vector to hold the angular flux values, indexed by boundary condition, polar angle, and group
        std::vector<std::vector<std::vector<double>>> _angflux;
        // Default constructor
        AngFluxBCFace() = default;
        // Constructor that initializes the angular flux to 0.0 with a specified size
        AngFluxBCFace(int nbc, int npol, int ng) {_resize_angflux(nbc, npol, ng, 0.0);};
        // Constructor that initializes the angular flux to a value with a specified size
        AngFluxBCFace(int nbc, int npol, int ng, double val) {_resize_angflux(nbc, npol, ng, val);};
    private:
        void _resize_angflux(int nbc, int npol, int ng, double val) {
            _angflux.resize(nbc);
            for (size_t i = 0; i < nbc; i++) {
                _angflux[i].resize(npol);
                for (size_t j = 0; j < npol; j++) {
                    _angflux[i][j].resize(ng, val);
                }
            }
        };
};

// Defines the angular flux boundary condition for a single angle
class AngFluxBCAngle
{
    public:
        // A vector of faces, each containing the angular flux for that face
        std::vector<AngFluxBCFace> _faces;
        // Default constructor
        AngFluxBCAngle() = default;
        // Constructor that initializes the faces vector with a specified number of faces
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

        // Constructor that initializes the LongRay object from a HighFive::Group
        // for a specific angle index and radians value
        LongRay(const HighFive::Group& group, int angle_index, double radians)
        : _radians(radians),
          _angle_index(angle_index),
          _fsrs(group.getDataSet("FSRs").read<std::vector<int>>()),
          _segments(group.getDataSet("Segments").read<std::vector<double>>()),
          _bc_face(group.getDataSet("BC_face").read<std::vector<int>>()),
          _bc_index(group.getDataSet("BC_index").read<std::vector<int>>())
          {
            // Read starting points, converting to pairs (since we don't care about the Z coordinate)
            auto points_data = group.getDataSet("Starting_Point").read<std::vector<double>>();
            _starting_points.reserve(points_data.size() / 3);
            for (int i = 0; i < points_data.size(); i += 3) {
                _starting_points.push_back(std::make_pair(points_data[i], points_data[i + 1]));
            };
            _bc_face[RAY_START] -= 1;
            _bc_face[RAY_END] -= 1;
            _bc_index[RAY_START] -= 1;
            _bc_index[RAY_END] -= 1;
        };
        int angle() const { return _angle_index; };
};
