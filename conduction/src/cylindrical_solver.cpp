#include "cylindrical_solver.hpp"
#include <limits>  // if not already included

auto is_invalid = [](double x) {
    return std::isnan(x) || std::isinf(x);
};

CylindricalSolver::CylindricalSolver(
    std::vector<std::shared_ptr<CylinderNode>> &nodes,
    const std::vector<std::shared_ptr<Solid>> &materials
) : nodes(nodes), materials(materials) {

    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        if (nodes[i]->get_outer_radius() > nodes[i + 1]->get_inner_radius()) {
            throw std::runtime_error("Invalid radius values for CylinderNode at index " + std::to_string(i));
        }
    }

    interface_temps.clear();
    interface_to_node_map.clear();
    node_to_interface_indices.clear();

    size_t interface_index = 0;

    // 🔹 Centerline: inner surface of fuel pellet
    interface_temps.push_back(600.0);
    interface_to_node_map.emplace_back(-1, 0);  // No node to the left of centerline
    size_t left_interface = interface_index++;

    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        double r_outer = nodes[i]->get_outer_radius();
        double r_next_inner = nodes[i + 1]->get_inner_radius();

        if (std::abs(r_outer - r_next_inner) > 1e-8) {
            // 🔹 Gap detected → two separate interface temps
            interface_temps.push_back(600.0); // outer of node[i]
            interface_to_node_map.emplace_back(i, -1); // right interface of node[i]
            size_t right_interface = interface_index++;

            node_to_interface_indices.emplace_back(left_interface, right_interface);

            interface_temps.push_back(600.0); // inner of node[i+1]
            interface_to_node_map.emplace_back(-1, i + 1); // left interface of node[i+1]
            left_interface = interface_index++; // for next loop
        } else {
            // 🔹 Solid-solid contact → shared interface
            interface_temps.push_back(600.0); // shared between node[i] and node[i+1]
            interface_to_node_map.emplace_back(i, i + 1);
            size_t right_interface = interface_index++;

            node_to_interface_indices.emplace_back(left_interface, right_interface);
            left_interface = right_interface;
        }
    }

    // 🔹 Last node (e.g., clad outer surface)
    interface_temps.push_back(0.0);
    interface_to_node_map.emplace_back(nodes.size() - 1, -1);
    size_t right_interface = interface_index++;

    node_to_interface_indices.emplace_back(left_interface, right_interface);
}

std::shared_ptr<CylinderNode> CylindricalSolver::get_node(const size_t index) {
    if (index >= nodes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return nodes[index];
}

std::vector<double> CylindricalSolver::get_volumes() const {
    std::vector<double> volumes;
    for (const auto &node : nodes) {
        volumes.push_back(node->get_volume());
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
        avg_temps[i] = nodes[i]->get_temperature();
    }
    return avg_temps;
}

std::vector<double> CylindricalSolver::solve(
    const std::vector<double>& qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations
) {
    std::vector<double> Tavg(nodes.size(), T_outer);
    std::vector<double> Tavg_prev(nodes.size(), T_outer);

    // Loop with dynamic gap conductance model until convergence
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        Tavg = _Tsolve(qdot, T_outer, tolerance, max_iterations);

        // Update radius values in the nodes based on the new calculated avg temperatures
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::shared_ptr<CylinderNode> node = nodes[i];
            std::shared_ptr<Solid> material = materials[i];

            double strain = material->strain(Tavg[i], Tavg_prev[i]);

            bool is_fuel = std::dynamic_pointer_cast<Fuel>(material) != nullptr;
            if (is_fuel) {
                node->set_inner_radius(node->get_inner_radius() * (1.0 + strain));
                node->set_outer_radius(node->get_outer_radius() * (1.0 + strain));
            } else {
                node->set_inner_radius(node->get_initial_inner_radius() * (1.0 + strain));
                node->set_outer_radius(node->get_initial_outer_radius() * (1.0 + strain));
            }
        }

        // Check convergence
        double max_diff = 0.0;
        for (size_t i = 0; i < nodes.size(); ++i) {
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

std::vector<double> CylindricalSolver::_Tsolve(
    const std::vector<double>& qdot,
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
            k[i] = materials[i]->k(nodes[i]->get_temperature());
        }

        interface_temps[interface_temps.size() - 1] = T_outer;
        double R_total = 0.0;

        for (size_t i = interface_temps.size(); i-- > 0;) {
            double R = 0.0;
            auto [left_node, right_node] = interface_to_node_map[i];

            if (left_node == -1 && right_node >= 0) {
                // Left boundary (e.g., centerline)
                interface_temps[i] = T_outer + qtotal * R_total;
            }

            if (left_node >= 0 && right_node >= 0) {
                // Assume this is a GAP between two solids (fuel/clad)
                R = 0.0;
            } else if (left_node >= 0) {
                // Internal to a node
                R = nodes[left_node]->calculate_thermal_resistance(k[left_node]);
            } else if (right_node >= 0) {
                R = nodes[right_node]->calculate_thermal_resistance(k[right_node]);
            }

            R_total += R;
            interface_temps[i] = T_outer + qtotal * R_total;
        }

        // Calculate node average temperatures
        for (size_t i = 0; i < nodes.size(); ++i) {

            // Calculate average temperature using 1-D heat conduction formula with 2 Dirichlet boundary conditions
            double R_inner = nodes[i]->get_inner_radius();
            double R_outer = nodes[i]->get_outer_radius();
            double R_avg = (R_inner + R_outer) * 0.5;
            // double T_inner = interface_temps[i];
            // double T_outer = interface_temps[i + 1];

            size_t idx_left = node_to_interface_indices[i].first;
            size_t idx_right = node_to_interface_indices[i].second;

            double T_inner = interface_temps[idx_left];
            double T_outer = interface_temps[idx_right];

            if (std::abs(R_inner) < 1e-9) {
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
                if (is_invalid(Tavg[i])) {
                    std::cerr << "NaN detected at node " << i << " with Tavg = " << Tavg[i]
                            << ", R_inner = " << R_inner << ", R_outer = " << R_outer
                            << ", qdot = " << qdot[i] << ", k = " << k[i] << std::endl;
                    throw std::runtime_error("Invalid temperature calculation at node " + std::to_string(i));
                }
            }
            nodes[i]->set_temperature(Tavg[i]); // Update node temperature
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
