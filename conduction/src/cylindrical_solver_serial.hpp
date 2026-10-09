#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "cylinder_node.hpp"
#include "materials.hpp"

/**
 * @brief Allocation-free building blocks shared by CylindricalSolverSerial and the batched
 *  per-pin loop in Conduction::solve_all_pins.
 *
 * Every function here works on caller-provided storage (raw arrays / containers that already
 * exist), so a solve performs no heap allocation. The node and material containers are template
 * parameters so the same code path (and therefore bit-identical arithmetic) is used for
 *   - std::vector<std::shared_ptr<CylinderNode>> / std::vector<std::shared_ptr<Solid>>
 *     (the public CylindricalSolverSerial API), and
 *   - plain stack arrays of CylinderNode / const Solid* (the hot loop over pins).
 */
namespace cylindrical_solver_detail {

inline CylinderNode& as_ref(CylinderNode& node) { return node; }
inline const CylinderNode& as_ref(const CylinderNode& node) { return node; }
inline CylinderNode& as_ref(const std::shared_ptr<CylinderNode>& node) { return *node; }
inline const Solid& as_ref(const Solid* material) { return *material; }
inline const Solid& as_ref(const std::shared_ptr<Solid>& material) { return *material; }

/**
 * @brief Scratch arrays for one solve, each of length num_nodes. Owned by the caller.
 */
struct Workspace {
    double* Tavg;           // [out] node average temperatures
    double* Tavg_prev;      // outer (radius update) loop previous iterate
    double* inner_Tavg_prev;// inner (temperature) loop previous iterate
    double* volumes;        // node volumes, frozen for one inner solve
    double* k;              // node thermal conductivities
    CylinderNode::LogRadii* logs; // log radii, frozen for one inner solve
};

/**
 * @brief Throw if any node overlaps the next one.
 */
template<typename Nodes>
void validate_radii(const Nodes& nodes, size_t num_nodes) {
    for (size_t i = 0; i < num_nodes - 1; ++i) {
        if (as_ref(nodes[i]).get_outer_radius() > as_ref(nodes[i + 1]).get_inner_radius()) {
            throw std::runtime_error("Invalid radius values for CylinderNode at index " + std::to_string(i));
        }
    }
}

/**
 * @brief Build the interface layout (one interface per solid-solid contact, two per gap).
 * @param node_to_interface [out] array of length num_nodes: (left, right) interface index per node.
 * @return Number of interfaces (at most 2 * num_nodes).
 */
template<typename Nodes>
size_t build_interface_layout(const Nodes& nodes, size_t num_nodes, std::pair<size_t, size_t>* node_to_interface) {
    size_t interface_index = 0;

    // innermost surface
    size_t left_interface = interface_index++;

    for (size_t i = 0; i < num_nodes - 1; ++i) {
        double r_outer = as_ref(nodes[i]).get_outer_radius();
        double r_next_inner = as_ref(nodes[i + 1]).get_inner_radius();

        if (std::abs(r_outer - r_next_inner) > 1e-8) {
            // Gap detected; two separate interface temps (outer of node[i], inner of node[i+1])
            size_t right_interface = interface_index++;
            node_to_interface[i] = {left_interface, right_interface};
            left_interface = interface_index++; // for next loop
        } else {
            // Solid-solid contact; shared interface (no gap)
            size_t right_interface = interface_index++;
            node_to_interface[i] = {left_interface, right_interface};
            left_interface = right_interface;
        }
    }

    // Last node (e.g., clad outer surface)
    size_t right_interface = interface_index++;
    node_to_interface[num_nodes - 1] = {left_interface, right_interface};

    return interface_index;
}

/**
 * @brief Fixed-geometry temperature solve (inner loop): iterate on k(T) with frozen node radii.
 *  Writes node average temperatures into ws.Tavg and interface temperatures into interface_temps.
 */
template<typename Nodes, typename Materials>
void internal_solve(
    Nodes& nodes,
    const Materials& materials,
    size_t N,
    const Fluid& gap_fluid,
    const std::pair<size_t, size_t>* node_to_interface,
    size_t num_interfaces,
    double* interface_temps,
    const double* qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations,
    const Workspace& ws
) {
    double* Tavg = ws.Tavg;
    double* Tavg_prev = ws.inner_Tavg_prev;
    double* volumes = ws.volumes;
    double* k = ws.k;
    CylinderNode::LogRadii* logs = ws.logs;

    // Radii do not change inside this loop, so radius-only quantities are computed once here
    for (size_t i = 0; i < N; ++i) {
        volumes[i] = as_ref(nodes[i]).get_volume();
        logs[i] = as_ref(nodes[i]).log_radii();
        Tavg_prev[i] = 0.0;
    }

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        // Calculate total heat generation
        double qtotal = 0.0;
        for (size_t i = 0; i < N; ++i) {
            qtotal += qdot[i] * volumes[i];
        }

        // Calculate thermal conductivity for each node
        for (size_t i = 0; i < N; ++i) {
            k[i] = as_ref(materials[i]).k(as_ref(nodes[i]).get_temperature());
        }

        // Calculate temperatures at cylinder node surfaces
        interface_temps[num_interfaces - 1] = T_outer;

        for (size_t i = N; i-- > 0;) {
            CylinderNode& node = as_ref(nodes[i]);
            auto [idx_left, idx_right] = node_to_interface[i];

            qtotal -= qdot[i] * volumes[i]; // Update total heat for next node
            double qflux = qtotal / node.get_inner_area(); // Inner heat flux coming into the node
            interface_temps[idx_left] = node.solve_inner_temperature(k[i], interface_temps[idx_right], qflux, qdot[i], logs[i]);

            if (i == 0) { continue; } // Skip the first node (no gap before the first node)

            // calculate gap width between this node and the next inward node
            const CylinderNode& inner_node = as_ref(nodes[i - 1]);
            double gap_width = node.get_inner_radius() - inner_node.get_outer_radius();
            if (gap_width > 1e-6) {

                // Gap conductance
                double k_gap = gap_fluid.k(interface_temps[idx_right]); // W/m-K
                double H_gap = k_gap / gap_width; // W/m^2-K

                // Add gap resistance to total resistance for the next node
                double R_gap = 1 / (H_gap * inner_node.get_outer_area());

                // Update the temperature at the outer surface of the next node
                interface_temps[idx_left - 1] = interface_temps[idx_left] + qtotal * R_gap;
            }
        }

        // Calculate node average temperatures
        for (size_t i = 0; i < N; ++i) {
            // Calculate average temperature using 1-D heat conduction formula with 2 Dirichlet boundary conditions
            CylinderNode& node = as_ref(nodes[i]);
            double T_inner = interface_temps[node_to_interface[i].first];
            double T_outer_node = interface_temps[node_to_interface[i].second];

            Tavg[i] = node.calculate_avg_temperature(k[i], T_inner, T_outer_node, qdot[i], logs[i]);
            node.set_temperature(Tavg[i]); // Update node temperature
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
        std::copy(Tavg, Tavg + N, Tavg_prev);
    }
}

/**
 * @brief Full pin solve (outer loop): alternate temperature solves with thermal-expansion radius
 *  updates until node average temperatures converge. Allocation-free.
 * @return true if converged within max_iterations.
 */
template<typename Nodes, typename Materials>
bool solve(
    Nodes& nodes,
    const Materials& materials,
    size_t N,
    const Fluid& gap_fluid,
    const std::pair<size_t, size_t>* node_to_interface,
    size_t num_interfaces,
    double* interface_temps,
    const double* qdot,
    double T_outer,
    double tolerance,
    size_t max_iterations,
    const Workspace& ws
) {
    double* Tavg = ws.Tavg;
    double* Tavg_prev = ws.Tavg_prev;

    // Initialize with T_outer
    std::fill(Tavg, Tavg + N, T_outer);
    std::fill(Tavg_prev, Tavg_prev + N, T_outer);

    // Loop with dynamic gap conductance model until convergence
    for (size_t iter = 0; iter < max_iterations; ++iter) {
        internal_solve(nodes, materials, N, gap_fluid, node_to_interface, num_interfaces,
                       interface_temps, qdot, T_outer, tolerance, max_iterations, ws);

        // Update radius values in the nodes based on the new calculated avg temperatures
        for (size_t i = 0; i < N; ++i) {
            CylinderNode& node = as_ref(nodes[i]);

            as_ref(materials[i]).update_node_radii(node, Tavg[i], Tavg_prev[i]);

            // Retrieve the outer radius of the previous node
            if (i > 0) {
                double r_outer_prev = as_ref(nodes[i - 1]).get_outer_radius();
                if (node.get_inner_radius() < r_outer_prev) {
                    double prev_volume = node.get_volume();

                    // Update inner radius to match the outer radius of the previous node
                    node.set_inner_radius(r_outer_prev);

                    // Adjust outer radius, conserving mass
                    double new_outer_radius = std::sqrt((prev_volume / (M_PI * node.get_height())) + (node.get_inner_radius() * node.get_inner_radius()));
                    node.set_outer_radius(new_outer_radius);
                }
            }
        }

        // Check convergence
        double max_diff = 0.0;
        for (size_t i = 0; i < N; ++i) {
            max_diff = std::max(max_diff, std::abs(Tavg[i] - Tavg_prev[i]));
        }
        if (max_diff < tolerance) {
            return true;
        }

        // Copy current to previous for next iteration
        std::copy(Tavg, Tavg + N, Tavg_prev);
    }
    return false;
}

} // namespace cylindrical_solver_detail

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

    // Scratch storage sized once at construction so solve() does not allocate per iteration
    std::vector<double> scratch;
    std::vector<CylinderNode::LogRadii> scratch_logs;

    void initialize_interface_data(double T_initial);
};
