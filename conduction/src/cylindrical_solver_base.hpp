#pragma once

#include <numeric>
#include <stdexcept>
#include <memory>
#include <Kokkos_Core.hpp>
#include "cylinder_node.hpp"
#include "materials.hpp"

template<typename ExecutionSpace>
class CylindricalSolverBase {
public:
    using execution_space = ExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    using DoubleView = Kokkos::View<double*, device_type>;
    using IntView = Kokkos::View<int*, device_type>;
    using PairView = Kokkos::View<Kokkos::pair<size_t, size_t>*, device_type>;

    CylindricalSolverBase(
        std::vector<std::shared_ptr<CylinderNode>> &nodes,
        const std::vector<std::shared_ptr<Solid>> &materials,
        double T_initial = 600.0 // Default initial temperature in Kelvin
    );

    virtual ~CylindricalSolverBase() = default;

    // Common interface methods
    double get_number_of_nodes() const { return nodes.size(); }
    void set_gap_fluid(const std::shared_ptr<Fluid> &fluid) { gap_fluid = fluid; }
    std::shared_ptr<CylinderNode> get_node(const size_t index);
    DoubleView get_volumes() const;
    DoubleView get_interface_temperatures() const;
    DoubleView get_average_temperatures() const;

    // Pure virtual method to be implemented by derived classes
    virtual DoubleView solve(
        const DoubleView &qdot,
        double T_outer,
        double tolerance = 1e-6,
        size_t max_iterations = 100
    ) = 0;

protected:
    bool is_solved = false;
    std::shared_ptr<Fluid> gap_fluid = std::make_shared<Helium>("He"); // Fluid for gap conductance model (default to Helium)
    std::vector<std::shared_ptr<CylinderNode>> nodes;
    std::vector<std::shared_ptr<Solid>> materials;
    DoubleView interface_temps;
    PairView node_to_interface_indices;

    // Common helper methods that can be used by derived classes
    void initialize_interface_data(double T_initial);
    DoubleView create_volumes_view() const;
    DoubleView create_average_temperatures_view() const;

    // Virtual method for execution space specific implementations
    virtual DoubleView internal_Tsolve(
        const DoubleView &qdot,
        double T_outer,
        double tolerance,
        size_t max_iterations
    ) = 0;
};