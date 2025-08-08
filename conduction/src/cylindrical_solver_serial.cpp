#include "cylindrical_solver_serial.hpp"

CylindricalSolverSerial::CylindricalSolverSerial(
    std::vector<std::shared_ptr<CylinderNode>>& nodes,
    const std::vector<std::shared_ptr<Solid>>& materials,
    double T_initial
) : nodes(nodes), materials(materials) {

    // Validate radii
    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        if (nodes[i]->get_outer_radius() > nodes[i + 1]->get_inner_radius()) {
            throw std::runtime_error("Invalid radius values for CylinderNode at index " + std::to_string(i));
        }
    }

    initialize_interface_data(T_initial);

    // Set initial temperatures for all nodes
    for (const auto& node : nodes) {
        node->set_temperature(T_initial);
    }
}

std::vector<double> CylindricalSolverSerial::solve(
    const std::vector<double>& qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations
) {
    std::vector<double> Tavg(nodes.size());
    std::vector<double> Tavg_prev(nodes.size());

    // Initialize with T_outer
    std::fill(Tavg.begin(), Tavg.end(), T_outer);
    std::fill(Tavg_prev.begin(), Tavg_prev.end(), T_outer);

    // Loop with dynamic gap conductance model until convergence
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        Tavg = internal_solve(qdot, T_outer, tolerance, max_iterations);

        // Update radius values in the nodes based on the new calculated avg temperatures
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::shared_ptr<CylinderNode> node = nodes[i];
            std::shared_ptr<Solid> material = materials[i];

            material->update_node_radii(node, Tavg[i], Tavg_prev[i]);

            // Retrieve the outer radius of the previous node
            if (i > 0) {
                double r_outer_prev = nodes[i - 1]->get_outer_radius();
                if (node->get_inner_radius() < r_outer_prev) {
                    double prev_volume = node->get_volume();

                    // Update inner radius to match the outer radius of the previous node
                    node->set_inner_radius(r_outer_prev);

                    // Adjust outer radius, conserving mass
                    double new_outer_radius = std::sqrt((prev_volume / (M_PI * node->get_height())) + (node->get_inner_radius() * node->get_inner_radius()));
                    node->set_outer_radius(new_outer_radius);
                }
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

        // Copy current to previous for next iteration
        Tavg_prev = Tavg;
    }

    if (!is_solved) {
        throw std::runtime_error("Cylindrical solver did not converge");
    }
    return Tavg;
}

std::vector<double> CylindricalSolverSerial::internal_solve(
    const std::vector<double>& qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations
) {
    const size_t N = nodes.size();

    std::vector<double> volumes = get_volumes();
    std::vector<double> Tavg(N, 0.0);
    std::vector<double> Tavg_prev(N, 0.0);
    std::vector<double> k(N);

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Calculate total heat generation
        double qtotal = 0.0;
        for (size_t i = 0; i < N; ++i) {
            qtotal += qdot[i] * volumes[i];
        }

        // Calculate thermal conductivity for each node
        for (size_t i = 0; i < N; ++i) {
            k[i] = materials[i]->k(nodes[i]->get_temperature());
        }

        // Calculate temperatures at cylinder node surfaces
        interface_temps[interface_temps.size() - 1] = T_outer;

        for (size_t i = nodes.size(); i-- > 0;) {
            auto [idx_left, idx_right] = node_to_interface_indices[i];

            qtotal -= qdot[i] * volumes[i]; // Update total heat for next node
            double qflux = qtotal / nodes[i]->get_inner_area(); // Inner heat flux coming into the node
            interface_temps[idx_left] = nodes[i]->solve_inner_temperature(k[i], interface_temps[idx_right], qflux, qdot[i]);

            if (i == 0) { continue; } // Skip the first node (no gap before the first node)

            // calculate gap width between this node and the next inward node
            double gap_width = nodes[i]->get_inner_radius() - nodes[i - 1]->get_outer_radius();
            if (gap_width > 1e-6) {

                // Gap conductance
                double k_gap = gap_fluid->k(interface_temps[idx_right]); // W/m-K
                double H_gap = k_gap / gap_width; // W/m^2-K

                // Add gap resistance to total resistance for the next node
                double R_gap = 1 / (H_gap * nodes[i - 1]->get_outer_area());

                // Update the temperature at the outer surface of the next node
                interface_temps[idx_left - 1] = interface_temps[idx_left] + qtotal * R_gap;
            }
        }

        // Calculate node average temperatures
        for (size_t i = 0; i < nodes.size(); ++i) {
            // Calculate average temperature using 1-D heat conduction formula with 2 Dirichlet boundary conditions
            size_t idx_left = node_to_interface_indices[i].first;
            size_t idx_right = node_to_interface_indices[i].second;

            double T_inner = interface_temps[idx_left];
            double T_outer_node = interface_temps[idx_right];

            Tavg[i] = nodes[i]->calculate_avg_temperature(k[i], T_inner, T_outer_node, qdot[i]);
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

        // Update previous temperatures for next iteration
        Tavg_prev = Tavg;
    }

    return Tavg;
}

std::vector<double> CylindricalSolverSerial::get_interface_temperatures() const {
    return interface_temps;
}

std::vector<double> CylindricalSolverSerial::get_volumes() const {
    std::vector<double> volumes;
    volumes.reserve(nodes.size());

    for (const auto& node : nodes) {
        volumes.push_back(node->get_volume());
    }

    return volumes;
}

void CylindricalSolverSerial::initialize_interface_data(double T_initial) {
    // Clear existing data
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
}
