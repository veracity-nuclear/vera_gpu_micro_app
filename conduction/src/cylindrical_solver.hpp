#pragma once

#include <numeric>
#include <stdexcept>
#include <memory>
#include "cylinder_node.hpp"
#include "materials.hpp"

class CylindricalSolver {
public:
    CylindricalSolver(std::vector<CylinderNode> &nodes);
    double get_number_of_nodes() const { return nodes.size(); }
    CylinderNode get_node(const size_t index);
    std::vector<double> get_volumes() const;
    std::vector<double> get_interface_temperatures() const;
    std::vector<double> get_average_temperatures() const;
    std::vector<double> solve_temperatures(
        const std::vector<double> &qdot,
        const std::vector<std::shared_ptr<SolidMaterial>> &materials,
        double T_outer,
        double tolerance = 1e-6,
        size_t max_iterations = 100
    );

private:
    bool is_solved = false;
    double T_outer;
    std::vector<double> qdot;
    std::vector<CylinderNode> nodes;
    std::vector<double> interface_temps;
};
