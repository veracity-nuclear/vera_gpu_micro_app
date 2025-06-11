#include "cylindrical_solver.hpp"
#include <numeric>
#include <stdexcept>

CylindricalSolver::CylindricalSolver(std::vector<CylinderNode> &nodes)
        : nodes(nodes), interface_temps(nodes.size() + 1, 0.0) {
        for (size_t i = 0; i < nodes.size() - 1; ++i) {
            if (std::abs(nodes[i].get_outer_radius() - nodes[i + 1].get_inner_radius()) > 1e-8) {
                throw std::runtime_error("Cylindrical nodes are not properly aligned");
            }
        }
    }

CylinderNode CylindricalSolver::get_node(const size_t index) {
    if (index >= nodes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return nodes[index];
}

std::vector<double> CylindricalSolver::get_volumes() const {
    std::vector<double> volumes;
    for (const auto &node : nodes) {
        volumes.push_back(node.get_volume());
    }
    return volumes;
}

std::vector<double> CylindricalSolver::get_interface_temperatures() const {
    if (interface_temps.empty()) {
        throw std::runtime_error("Interface temperatures have not been calculated yet");
    }
    return interface_temps;
}

std::vector<double> CylindricalSolver::get_average_temperatures() const {
    if (interface_temps.empty()) {
        throw std::runtime_error("Interface temperatures have not been calculated yet");
    }
    std::vector<double> Tavg(nodes.size(), 0.0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        Tavg[i] = (interface_temps[i] + interface_temps[i + 1]) / 2.0;
    }
    return Tavg;
}

std::vector<double> CylindricalSolver::solve_temperatures(
    const std::vector<double>& qdot,
    const std::vector<double>& k,
    double T_outer
) {
    const size_t N = nodes.size();
    if (qdot.size() != N || k.size() != N) {
        throw std::invalid_argument("Mismatch between input sizes and number of nodes");
    }

    std::vector<double> volumes = get_volumes();
    double qtotal = std::inner_product(qdot.begin(), qdot.end(), volumes.begin(), 0.0);

    double R_total = 0.0;
    interface_temps[N] = T_outer; // Outer boundary condition

    for (size_t i = N; i-- > 0;) {
        R_total += nodes[i].calculate_thermal_resistance(k[i]);
        interface_temps[i] = T_outer + qtotal * R_total;
    }

    return interface_temps;
}
