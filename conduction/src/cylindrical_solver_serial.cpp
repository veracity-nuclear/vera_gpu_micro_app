#include "cylindrical_solver_serial.hpp"

CylindricalSolverSerial::CylindricalSolverSerial(
    std::vector<std::shared_ptr<CylinderNode>>& nodes,
    const std::vector<std::shared_ptr<Solid>>& materials,
    double T_initial
) : nodes(nodes), materials(materials) {

    // Validate radii
    cylindrical_solver_detail::validate_radii(nodes, nodes.size());

    initialize_interface_data(T_initial);

    // Set initial temperatures for all nodes
    for (const auto& node : nodes) {
        node->set_temperature(T_initial);
    }

    // Workspace: Tavg, Tavg_prev, inner_Tavg_prev, volumes, k, logs (each nodes.size() long)
    scratch.resize(5 * nodes.size());
    scratch_logs.resize(nodes.size());
}

std::vector<double> CylindricalSolverSerial::solve(
    const std::vector<double>& qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations
) {
    const size_t N = nodes.size();
    double* s = scratch.data();
    cylindrical_solver_detail::Workspace ws{s, s + N, s + 2 * N, s + 3 * N, s + 4 * N, scratch_logs.data()};

    bool converged = cylindrical_solver_detail::solve(
        nodes, materials, N, *gap_fluid,
        node_to_interface_indices.data(), interface_temps.size(), interface_temps.data(),
        qdot.data(), T_outer, tolerance, max_iterations, ws
    );
    if (converged) {
        is_solved = true;
    }

    if (!is_solved) {
        throw std::runtime_error("Cylindrical solver did not converge");
    }
    return std::vector<double>(ws.Tavg, ws.Tavg + N);
}

std::vector<double> CylindricalSolverSerial::get_interface_temperatures() const {
    return interface_temps;
}

void CylindricalSolverSerial::initialize_interface_data(double T_initial) {
    node_to_interface_indices.resize(nodes.size());
    size_t num_interfaces = cylindrical_solver_detail::build_interface_layout(
        nodes, nodes.size(), node_to_interface_indices.data());
    interface_temps.assign(num_interfaces, T_initial);
}
