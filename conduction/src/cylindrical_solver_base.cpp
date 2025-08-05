#include "cylindrical_solver_base.hpp"

// Template implementations

template<typename ExecutionSpace>
CylindricalSolverBase<ExecutionSpace>::CylindricalSolverBase(
    std::vector<std::shared_ptr<CylinderNode>> &nodes,
    const std::vector<std::shared_ptr<Solid>> &materials,
    double T_initial
) : nodes(nodes), materials(materials) {

    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        if (nodes[i]->get_outer_radius() > nodes[i + 1]->get_inner_radius()) {
            throw std::runtime_error("Invalid radius values for CylinderNode at index " + std::to_string(i));
        }
    }

    initialize_interface_data(T_initial);

    // Set initial temperatures for all nodes
    for (const auto &node : nodes) {
        node->set_temperature(T_initial);
    }
}

template<typename ExecutionSpace>
std::shared_ptr<CylinderNode> CylindricalSolverBase<ExecutionSpace>::get_node(const size_t index) {
    if (index >= nodes.size()) {
        throw std::out_of_range("Index out of range");
    }
    return nodes[index];
}

template<typename ExecutionSpace>
typename CylindricalSolverBase<ExecutionSpace>::DoubleView
CylindricalSolverBase<ExecutionSpace>::get_volumes() const {
    return create_volumes_view();
}

template<typename ExecutionSpace>
typename CylindricalSolverBase<ExecutionSpace>::DoubleView
CylindricalSolverBase<ExecutionSpace>::get_interface_temperatures() const {
    if (interface_temps.extent(0) == 0) {
        throw std::runtime_error("Interface temperatures have not been calculated yet");
    }
    return interface_temps;
}

template<typename ExecutionSpace>
typename CylindricalSolverBase<ExecutionSpace>::DoubleView
CylindricalSolverBase<ExecutionSpace>::get_average_temperatures() const {
    return create_average_temperatures_view();
}

template<typename ExecutionSpace>
void CylindricalSolverBase<ExecutionSpace>::initialize_interface_data(double T_initial) {
    // Temporary vectors for construction
    std::vector<double> temp_interface_temps;
    std::vector<std::pair<size_t, size_t>> temp_node_to_interface_indices;

    size_t interface_index = 0;

    // innermost surface
    temp_interface_temps.push_back(T_initial);
    size_t left_interface = interface_index++;

    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        double r_outer = nodes[i]->get_outer_radius();
        double r_next_inner = nodes[i + 1]->get_inner_radius();

        if (std::abs(r_outer - r_next_inner) > 1e-8) {
            // Gap detected; two separate interface temps
            temp_interface_temps.push_back(T_initial); // outer of node[i]
            size_t right_interface = interface_index++;

            temp_node_to_interface_indices.emplace_back(left_interface, right_interface);

            temp_interface_temps.push_back(T_initial); // inner of node[i+1]
            left_interface = interface_index++; // for next loop
        } else {
            // Solid-solid contact; shared interface (no gap)
            temp_interface_temps.push_back(T_initial); // shared between node[i] and node[i+1]
            size_t right_interface = interface_index++;

            temp_node_to_interface_indices.emplace_back(left_interface, right_interface);
            left_interface = right_interface;
        }
    }

    // Last node (e.g., clad outer surface)
    temp_interface_temps.push_back(T_initial);
    size_t right_interface = interface_index++;

    temp_node_to_interface_indices.emplace_back(left_interface, right_interface);

    // Initialize Kokkos::Views
    interface_temps = DoubleView("interface_temps", temp_interface_temps.size());
    node_to_interface_indices = PairView("node_to_interface_indices", temp_node_to_interface_indices.size());

    // Copy data to Kokkos::Views
    auto h_interface_temps = Kokkos::create_mirror_view(interface_temps);
    auto h_node_to_interface_indices = Kokkos::create_mirror_view(node_to_interface_indices);

    for (size_t i = 0; i < temp_interface_temps.size(); ++i) {
        h_interface_temps(i) = temp_interface_temps[i];
    }
    for (size_t i = 0; i < temp_node_to_interface_indices.size(); ++i) {
        h_node_to_interface_indices(i) = Kokkos::make_pair(temp_node_to_interface_indices[i].first,
                                                          temp_node_to_interface_indices[i].second);
    }

    Kokkos::deep_copy(interface_temps, h_interface_temps);
    Kokkos::deep_copy(node_to_interface_indices, h_node_to_interface_indices);
}

template<typename ExecutionSpace>
typename CylindricalSolverBase<ExecutionSpace>::DoubleView
CylindricalSolverBase<ExecutionSpace>::create_volumes_view() const {
    DoubleView volumes("volumes", nodes.size());
    auto h_volumes = Kokkos::create_mirror_view(volumes);

    for (size_t i = 0; i < nodes.size(); ++i) {
        h_volumes(i) = nodes[i]->get_volume();
    }

    Kokkos::deep_copy(volumes, h_volumes);
    return volumes;
}

template<typename ExecutionSpace>
typename CylindricalSolverBase<ExecutionSpace>::DoubleView
CylindricalSolverBase<ExecutionSpace>::create_average_temperatures_view() const {
    DoubleView avg_temps("avg_temps", nodes.size());
    auto h_avg_temps = Kokkos::create_mirror_view(avg_temps);

    for (size_t i = 0; i < nodes.size(); ++i) {
        h_avg_temps(i) = nodes[i]->get_temperature();
    }

    Kokkos::deep_copy(avg_temps, h_avg_temps);
    return avg_temps;
}

// Explicit template instantiations for the execution spaces we support
template class CylindricalSolverBase<Kokkos::Serial>;

#ifdef KOKKOS_ENABLE_OPENMP
template class CylindricalSolverBase<Kokkos::OpenMP>;
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class CylindricalSolverBase<Kokkos::Cuda>;
#endif
