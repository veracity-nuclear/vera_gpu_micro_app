#pragma once

#include "cylinder_node.hpp"

class CylindricalSolver {
public:
    CylindricalSolver(std::vector<CylinderNode> &nodes);
    double get_number_of_nodes() const { return nodes.size(); }
    CylinderNode get_node(const size_t index);
    std::vector<double> get_volumes() const;
    std::vector<double> get_interface_temperatures() const;
    std::vector<double> get_average_temperatures() const;
    std::vector<double> solve_temperatures(const std::vector<double> &qdot, const std::vector<double> &k, double T_outer);

private:
    std::vector<CylinderNode> nodes;
    std::vector<double> interface_temps;
};
