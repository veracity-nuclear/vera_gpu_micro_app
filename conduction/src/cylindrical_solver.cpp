#include "cylindrical_solver.hpp"

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
    std::vector<double> avg_temps(nodes.size(), 0.0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        avg_temps[i] = nodes[i].get_temperature();
    }
    return avg_temps;
}

std::vector<double> CylindricalSolver::solve_temperatures(
    const std::vector<double>& qdot,
    const std::vector<std::shared_ptr<SolidMaterial>>& materials,
    double T_outer,
    double tolerance,
    size_t max_iterations
) {
    const size_t N = nodes.size();

    std::vector<double> volumes = get_volumes();
    double qtotal = std::inner_product(qdot.begin(), qdot.end(), volumes.begin(), 0.0);

    std::vector<double> Tavg(nodes.size(), 0.0);
    std::vector<double> Tavg_prev(nodes.size(), 0.0);
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Calculate thermal conductivity for each node
        std::vector<double> k(N);
        for (size_t i = 0; i < N; ++i) {
            k[i] = materials[i]->k(nodes[i].get_temperature());
        }

        // Calculate interface temperatures
        double R_total = 0.0;
        interface_temps[N] = T_outer; // Outer boundary condition

        for (size_t i = N; i-- > 0;) {
            R_total += nodes[i].calculate_thermal_resistance(k[i]);
            interface_temps[i] = T_outer + qtotal * R_total;
        }

        // Calculate node average temperatures
        for (size_t i = 0; i < nodes.size(); ++i) {

            // Calculate average temperature using 1-D heat conduction formula with 2 Dirichlet boundary conditions
            double R_inner = nodes[i].get_inner_radius();
            double R_outer = nodes[i].get_outer_radius();
            double R_avg = (R_inner + R_outer) * 0.5;
            double T_inner = interface_temps[i];
            double T_outer = interface_temps[i + 1];

            if (R_inner == 0.0) {
                // Special case for the center of a cylinder (R_inner = 0)
                Tavg[i] = T_outer + 3.0 / 16.0 * (qdot[i] / k[i]) * R_outer * R_outer;
            } else {
                // Solve for C1
                double num = (T_outer - T_inner) + (qdot[i] / (4.0 * k[i])) * (R_outer * R_outer - R_inner * R_inner);
                double denom = std::log(R_outer) - std::log(R_inner);
                double C1 = num / denom;

                // Solve for C2
                double C2 = T_inner + (qdot[i] / (4.0 * k[i])) * R_inner * R_inner - C1 * std::log(R_inner);

                // Evaluate T at average radius
                Tavg[i] = -(qdot[i] / (4.0 * k[i])) * R_avg * R_avg + C1 * std::log(R_avg) + C2;
            }
            nodes[i].set_temperature(Tavg[i]); // Update node temperature
        }
        // Check convergence
        double max_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
            max_diff = std::max(max_diff, std::abs(Tavg[i] - Tavg_prev[i]));
        }
        if (max_diff < tolerance) {
            is_solved = true;
            break;
        }
        Tavg_prev = Tavg; // Update previous temperatures for next iteration
    }
    if (!is_solved) {
        throw std::runtime_error("Cylindrical solver did not converge");
    }
    return Tavg;
}
