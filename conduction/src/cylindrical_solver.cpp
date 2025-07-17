#include "cylindrical_solver.hpp"

CylindricalSolver::CylindricalSolver(
    std::vector<std::shared_ptr<CylinderNode>> &nodes,
    const std::vector<std::shared_ptr<Solid>> &materials,
    double T_initial
) : nodes(nodes), materials(materials) {

    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        if (nodes[i]->get_outer_radius() > nodes[i + 1]->get_inner_radius()) {
            throw std::runtime_error("Invalid radius values for CylinderNode at index " + std::to_string(i));
        }
    }

    interface_temps.clear();
    node_to_interface_indices.clear();

    size_t interface_index = 0;

    // innermost surface
    interface_temps.push_back(T_initial);
    size_t left_interface = interface_index++;

    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        double r_outer = nodes[i]->get_outer_radius();
        double r_next_inner = nodes[i + 1]->get_inner_radius();

        if (std::abs(r_outer - r_next_inner) > 1e-8) {
            // Gap detected; two separate interface temps
            interface_temps.push_back(T_initial); // outer of node[i]
            size_t right_interface = interface_index++;

            node_to_interface_indices.emplace_back(left_interface, right_interface);

            interface_temps.push_back(T_initial); // inner of node[i+1]
            left_interface = interface_index++; // for next loop
        } else {
            // Solid-solid contact; shared interface (no gap)
            interface_temps.push_back(T_initial); // shared between node[i] and node[i+1]
            size_t right_interface = interface_index++;

            node_to_interface_indices.emplace_back(left_interface, right_interface);
            left_interface = right_interface;
        }
    }

    // Last node (e.g., clad outer surface)
    interface_temps.push_back(T_initial);
    size_t right_interface = interface_index++;

    node_to_interface_indices.emplace_back(left_interface, right_interface);

    // Set initial temperatures for all nodes
    for (const auto &node : nodes) {
        node->set_temperature(T_initial);
    }
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

            material->update_node_radii(node, Tavg[i], Tavg_prev[i]);
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

        // Calculate temperatures at cylinder node surfaces
        interface_temps[interface_temps.size() - 1] = T_outer;
        double R_total = 0.0;

        for (size_t i = nodes.size(); i-- > 0;) {
            auto [idx_left, idx_right] = node_to_interface_indices[i];

            interface_temps[idx_right] = T_outer + qtotal * R_total;
            R_total += nodes[i]->calculate_thermal_resistance(k[i]); // W/m-K
            interface_temps[idx_left] = T_outer + qtotal * R_total;

            if (i == 0) { continue; } // Skip the first node (no gap before the first node)

            // calculate gap width between this node and the next inward node
            double gap_width = nodes[i]->get_inner_radius() - nodes[i - 1]->get_outer_radius();
            if (gap_width > 0.0) {
                // Assume Helium is the fill gas in the gap for now
                double k_gap = gap_fluid->k(interface_temps[idx_right]); // W/m-K

                // Gap conductance
                double H_gap = k_gap / gap_width; // W/m^2-K

                // Add gap resistance to total resistance for the next node
                R_total += 1 / (H_gap * nodes[i]->get_inner_area()); // W/m-K
            }
        }

        // Calculate node average temperatures
        for (size_t i = 0; i < nodes.size(); ++i) {

            // Calculate average temperature using 1-D heat conduction formula with 2 Dirichlet boundary conditions
            double R_inner = nodes[i]->get_inner_radius();
            double R_outer = nodes[i]->get_outer_radius();
            double R_avg = (R_inner + R_outer) * 0.5;

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
            }
            nodes[i]->set_temperature(Tavg[i]); // Update node temperature
        }

        // Check convergence
        double max_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
            max_diff = std::max(max_diff, std::abs(Tavg[i] - Tavg_prev[i]));
        }
        if (max_diff < tolerance) {
            break;
        }
        Tavg_prev = Tavg; // Update previous temperatures for next iteration
    }
    return Tavg;
}
