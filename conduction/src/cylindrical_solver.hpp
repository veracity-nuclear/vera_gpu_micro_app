#pragma once

#include <numeric>
#include <stdexcept>
#include <memory>
#include "cylinder_node.hpp"
#include "materials.hpp"

class CylindricalSolver {
public:
    CylindricalSolver(
        std::vector<std::shared_ptr<CylinderNode>> &nodes,
        const std::vector<std::shared_ptr<Solid>> &materials
    );
    double get_number_of_nodes() const { return nodes.size(); }
    std::shared_ptr<CylinderNode> get_node(const size_t index);
    std::vector<double> get_volumes() const;
    std::vector<double> get_interface_temperatures() const;
    std::vector<double> get_average_temperatures() const;
    std::vector<double> solve(
        const std::vector<double> &qdot,
        double T_outer,
        double tolerance = 1e-6,
        size_t max_iterations = 100
    );

private:
    bool is_solved = false;
    std::vector<std::shared_ptr<CylinderNode>> nodes;
    std::vector<std::shared_ptr<Solid>> materials;
    std::vector<double> interface_temps;

    // Each pair is (left_node_index, right_node_index)
    // - If either is -1, it's a boundary
    std::vector<std::pair<int, int>> interface_to_node_map;

    // For each node, store (left_interface_idx, right_interface_idx)
    std::vector<std::pair<size_t, size_t>> node_to_interface_indices;

    std::vector<double> _Tsolve(
        const std::vector<double> &qdot,
        double T_outer,
        double tolerance,
        size_t max_iterations
    );
};
