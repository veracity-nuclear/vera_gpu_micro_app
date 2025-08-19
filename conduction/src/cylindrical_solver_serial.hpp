#pragma once

#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "cylinder_node.hpp"
#include "materials.hpp"

class CylindricalSolverSerial {
public:
    CylindricalSolverSerial(
        std::vector<std::shared_ptr<CylinderNode>>& nodes,
        const std::vector<std::shared_ptr<Solid>>& materials,
        double T_initial = 600.0
    );

    ~CylindricalSolverSerial() = default;

    // Solve for temperature distribution
    std::vector<double> solve(
        const std::vector<double>& qdot,
        double T_outer,
        double tolerance = 1e-6,
        size_t max_iterations = 100
    );

    // Get interface temperatures
    std::vector<double> get_interface_temperatures() const;

    // Set gap fluid for conductance calculations
    void set_gap_fluid(const std::shared_ptr<Fluid>& fluid) { gap_fluid = fluid; }

    // Get number of nodes
    size_t get_number_of_nodes() const { return nodes.size(); }

private:
    bool is_solved = false;
    std::shared_ptr<Fluid> gap_fluid = std::make_shared<Helium>("He");
    std::vector<std::shared_ptr<CylinderNode>> nodes;
    std::vector<std::shared_ptr<Solid>> materials;
    std::vector<double> interface_temps;
    std::vector<std::pair<size_t, size_t>> node_to_interface_indices;

    void initialize_interface_data(double T_initial);
    std::vector<double> get_volumes() const;

    std::vector<double> internal_solve(
        const std::vector<double>& qdot,
        double T_outer,
        double tolerance,
        size_t max_iterations
    );
};
