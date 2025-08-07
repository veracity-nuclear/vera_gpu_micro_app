#include "cylindrical_solver_serial.hpp"

CylindricalSolverSerial::DoubleView1D CylindricalSolverSerial::solve(
    const DoubleView1D& qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations
) {
    DoubleView1D Tavg("Tavg", nodes.size());
    DoubleView1D Tavg_prev("Tavg_prev", nodes.size());

    // Initialize with T_outer
    auto h_Tavg = Kokkos::create_mirror_view(Tavg);
    auto h_Tavg_prev = Kokkos::create_mirror_view(Tavg_prev);

    for (size_t i = 0; i < nodes.size(); ++i) {
        h_Tavg(i) = T_outer;
        h_Tavg_prev(i) = T_outer;
    }

    Kokkos::deep_copy(Tavg, h_Tavg);
    Kokkos::deep_copy(Tavg_prev, h_Tavg_prev);

    // Loop with dynamic gap conductance model until convergence
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        Tavg = internal_Tsolve(qdot, T_outer, tolerance, max_iterations);

        // Copy current to host for node updates
        Kokkos::deep_copy(h_Tavg, Tavg);
        Kokkos::deep_copy(h_Tavg_prev, Tavg_prev);

        // Update radius values in the nodes based on the new calculated avg temperatures
        for (size_t i = 0; i < nodes.size(); ++i) {
            std::shared_ptr<CylinderNode> node = nodes[i];
            std::shared_ptr<Solid> material = materials[i];

            material->update_node_radii(node, h_Tavg(i), h_Tavg_prev(i));

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
            max_diff = std::max(max_diff, std::abs(h_Tavg(i) - h_Tavg_prev(i)));
        }
        if (max_diff < tolerance) {
            is_solved = true;
            break;
        }

        // Copy current to previous for next iteration
        Kokkos::deep_copy(Tavg_prev, Tavg);
    }
    if (!is_solved) {
        throw std::runtime_error("Cylindrical solver did not converge");
    }
    return Tavg;
}

CylindricalSolverSerial::DoubleView1D CylindricalSolverSerial::internal_Tsolve(
    const DoubleView1D& qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations
) {
    const size_t N = nodes.size();

    DoubleView1D volumes = get_volumes();
    DoubleView1D Tavg("Tavg", N);
    DoubleView1D Tavg_prev("Tavg_prev", N);
    DoubleView1D k("k", N);

    // Initialize to zero
    Kokkos::deep_copy(Tavg, 0.0);
    Kokkos::deep_copy(Tavg_prev, 0.0);

    // Create mirror views for host access
    auto h_volumes = Kokkos::create_mirror_view(volumes);
    auto h_Tavg = Kokkos::create_mirror_view(Tavg);
    auto h_Tavg_prev = Kokkos::create_mirror_view(Tavg_prev);
    auto h_k = Kokkos::create_mirror_view(k);
    auto h_qdot = Kokkos::create_mirror_view(qdot);
    auto h_interface_temps = Kokkos::create_mirror_view(interface_temps);
    auto h_node_to_interface_indices = Kokkos::create_mirror_view(node_to_interface_indices);

    // Copy views to host
    Kokkos::deep_copy(h_volumes, volumes);
    Kokkos::deep_copy(h_qdot, qdot);
    Kokkos::deep_copy(h_interface_temps, interface_temps);
    Kokkos::deep_copy(h_node_to_interface_indices, node_to_interface_indices);

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Calculate total heat generation
        double qtotal = 0.0;
        for (size_t i = 0; i < N; ++i) {
            qtotal += h_qdot(i) * h_volumes(i);
        }

        // Calculate thermal conductivity for each node (Serial execution - no parallelization needed)
        for (size_t i = 0; i < N; ++i) {
            h_k(i) = materials[i]->k(nodes[i]->get_temperature());
        }

        // Calculate temperatures at cylinder node surfaces (Serial execution)
        h_interface_temps(interface_temps.extent(0) - 1) = T_outer;

        for (size_t i = nodes.size(); i-- > 0;) {
            auto [idx_left, idx_right] = h_node_to_interface_indices(i);

            qtotal -= h_qdot(i) * h_volumes(i); // Update total heat for next node
            double qflux = qtotal / nodes[i]->get_inner_area(); // Inner heat flux coming into the node
            h_interface_temps(idx_left) = nodes[i]->solve_inner_temperature(h_k(i), h_interface_temps(idx_right), qflux, h_qdot(i));

            if (i == 0) { continue; } // Skip the first node (no gap before the first node)

            // calculate gap width between this node and the next inward node
            double gap_width = nodes[i]->get_inner_radius() - nodes[i - 1]->get_outer_radius();
            if (gap_width > 1e-6) {

                // Gap conductance
                double k_gap = gap_fluid->k(h_interface_temps(idx_right)); // W/m-K
                double H_gap = k_gap / gap_width; // W/m^2-K

                // Add gap resistance to total resistance for the next node
                double R_gap = 1 / (H_gap * nodes[i - 1]->get_outer_area());

                // Update the temperature at the outer surface of the next node
                h_interface_temps(idx_left - 1) = h_interface_temps(idx_left) + qtotal * R_gap;
            }
        }

        // Calculate node average temperatures (Serial execution)
        for (size_t i = 0; i < nodes.size(); ++i) {
            // Calculate average temperature using 1-D heat conduction formula with 2 Dirichlet boundary conditions
            size_t idx_left = h_node_to_interface_indices(i).first;
            size_t idx_right = h_node_to_interface_indices(i).second;

            double T_inner = h_interface_temps(idx_left);
            double T_outer = h_interface_temps(idx_right);

            h_Tavg(i) = nodes[i]->calculate_avg_temperature(h_k(i), T_inner, T_outer, h_qdot(i));
            nodes[i]->set_temperature(h_Tavg(i)); // Update node temperature
        }

        // Check convergence
        double max_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
            max_diff = std::max(max_diff, std::abs(h_Tavg(i) - h_Tavg_prev(i)));
        }
        if (max_diff < tolerance) {
            break;
        }

        // Update previous temperatures for next iteration
        for (size_t i = 0; i < N; ++i) {
            h_Tavg_prev(i) = h_Tavg(i);
        }
    }

    // Copy results back to device
    Kokkos::deep_copy(Tavg, h_Tavg);
    Kokkos::deep_copy(interface_temps, h_interface_temps);

    return Tavg;
}
