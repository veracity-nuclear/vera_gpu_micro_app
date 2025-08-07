#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <numeric>
#include <iostream>
#include <cassert>
#include <Kokkos_Core.hpp>
#include "cylinder_node.hpp"
#include "cylindrical_solver_base.hpp"
#include "cylindrical_solver_serial.hpp"
#include "materials.hpp"
#include "hdf5_utils.hpp"

template<typename ExecutionSpace>
class Conduction {
public:
    using execution_space = ExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    using DoubleView1D = Kokkos::View<double*, device_type>;
    using DoubleView2D = Kokkos::View<double**, device_type>;

    struct ConductionResults {
        std::vector<double> average_temperatures;
        std::vector<std::vector<double>> interface_temperatures;
        double elapsed_time_ms;
        size_t total_pins;
    };

    Conduction(
        const std::vector<double>& radii,
        double height,
        const std::vector<std::shared_ptr<Solid>>& materials
    );

    // Load data from HDF5 file
    void load_pin_data(
        const std::string& filename,
        const std::vector<std::string>& state_groups
    );

    // Solve all pins in parallel
    ConductionResults solve_all_pins();

private:
    std::vector<double> radii_;
    double height_;
    std::vector<std::shared_ptr<Solid>> materials_;

    // Pin data loaded from HDF5
    std::vector<double> pin_powers_;
    std::vector<double> clad_surf_temps_;
    bool data_loaded_ = false;
};

// Template implementations

template<typename ExecutionSpace>
Conduction<ExecutionSpace>::Conduction(
    const std::vector<double>& radii,
    double height,
    const std::vector<std::shared_ptr<Solid>>& materials
) : radii_(radii), height_(height), materials_(materials) {
    if (radii_.size() != 4 || materials_.size() != 2) {
        throw std::invalid_argument("Expected 4 radii and 2 materials for fuel-gap-clad geometry");
    }
}

template<typename ExecutionSpace>
void Conduction<ExecutionSpace>::load_pin_data(
    const std::string& filename,
    const std::vector<std::string>& state_groups
) {
    pin_powers_.clear();
    clad_surf_temps_.clear();

    for (const auto& group : state_groups) {
        double total_power = read_hdf5_scalar(filename, group + "/total_power");
        double power_percent = read_hdf5_scalar(filename, group + "/power") * 0.01; // convert to fraction

        FlatHDF5Data pin_powers = read_flat_hdf5_dataset(filename, group + "/pin_powers");
        double scaling = total_power * power_percent / std::accumulate(pin_powers.data.begin(), pin_powers.data.end(), 0.0);
        for (auto& p : pin_powers.data) p *= scaling;

        FlatHDF5Data clad_surf_temps = read_flat_hdf5_dataset(filename, group + "/pin_max_clad_surface_temp");
        for (auto& T : clad_surf_temps.data) T += 273.15;  // convert to K

        // Append to member vectors
        pin_powers_.insert(pin_powers_.end(), pin_powers.data.begin(), pin_powers.data.end());
        clad_surf_temps_.insert(clad_surf_temps_.end(), clad_surf_temps.data.begin(), clad_surf_temps.data.end());
    }

    data_loaded_ = true;
}

template<typename ExecutionSpace>
typename Conduction<ExecutionSpace>::ConductionResults
Conduction<ExecutionSpace>::solve_all_pins() {
    if (!data_loaded_) {
        throw std::runtime_error("Pin data not loaded. Call load_pin_data() first.");
    }

    assert(pin_powers_.size() == clad_surf_temps_.size());
    size_t N = pin_powers_.size();

    // Determine the number of nodes dynamically
    std::vector<std::shared_ptr<CylinderNode>> sample_nodes = {
        std::make_shared<CylinderNode>(height_, radii_[0], radii_[1]), // fuel node
        std::make_shared<CylinderNode>(height_, radii_[2], radii_[3])  // clad node
    };
    size_t num_nodes = sample_nodes.size();

    ConductionResults results;
    results.total_pins = N;
    results.average_temperatures.resize(N * num_nodes); // dynamic number of nodes per pin
    results.interface_temperatures.resize(N); // N pins, each with multiple interfaces

    auto start = std::chrono::high_resolution_clock::now();

    // Prepare data for parallel execution
    DoubleView1D pin_powers_view("pin_powers", N);
    DoubleView1D clad_temps_view("clad_temps", N);
    DoubleView2D results_temps("results_temps", N, num_nodes); // dynamic number of nodes per pin
    DoubleView2D interface_temps("interface_temps", N, num_nodes + 1); // num_nodes + 1 interfaces per pin

    // Copy data to device
    auto h_pin_powers = Kokkos::create_mirror_view(pin_powers_view);
    auto h_clad_temps = Kokkos::create_mirror_view(clad_temps_view);

    for (size_t i = 0; i < N; ++i) {
        h_pin_powers(i) = pin_powers_[i];
        h_clad_temps(i) = clad_surf_temps_[i];
    }

    Kokkos::deep_copy(pin_powers_view, h_pin_powers);
    Kokkos::deep_copy(clad_temps_view, h_clad_temps);

    // For now, we'll use a simple loop since the solver creation requires host-side operations
    // TODO: Refactor solvers to be device-compatible for full parallel execution
    for (size_t i = 0; i < N; ++i) {
        // Create nodes for this pin (fuel and clad with gap between)
        std::vector<std::shared_ptr<CylinderNode>> nodes = {
            std::make_shared<CylinderNode>(height_, radii_[0], radii_[1]), // fuel node
            std::make_shared<CylinderNode>(height_, radii_[2], radii_[3])  // clad node
        };

        // Create solver for this pin using the appropriate concrete class
        std::unique_ptr<CylindricalSolverBase<ExecutionSpace>> solver;

        if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Serial>) {
            solver = std::make_unique<CylindricalSolverSerial>(nodes, materials_);
        }
#ifdef KOKKOS_ENABLE_OPENMP
        else if constexpr (std::is_same_v<ExecutionSpace, Kokkos::OpenMP>) {
            // TODO: Create CylindricalSolverOpenMP when it exists
            throw std::runtime_error("OpenMP solver not yet implemented");
        }
#endif
#ifdef KOKKOS_ENABLE_CUDA
        else if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
            // TODO: Create CylindricalSolverCuda when it exists
            throw std::runtime_error("CUDA solver not yet implemented");
        }
#endif
        else {
            throw std::runtime_error("Unsupported execution space");
        }

        double T_outer = clad_surf_temps_[i];

        // Create Kokkos::View for qdot
        typename CylindricalSolverBase<ExecutionSpace>::DoubleView1D qdot("qdot", nodes.size());
        auto h_qdot = Kokkos::create_mirror_view(qdot);

        for (size_t j = 0; j < nodes.size(); ++j) {
            h_qdot(j) = 0.0;
        }
        h_qdot(0) = pin_powers_[i] / nodes[0]->get_volume(); // only fuel node has heat generation
        Kokkos::deep_copy(qdot, h_qdot);

        // Solve for this pin
        auto Tavg = solver->solve(qdot, T_outer);

        // Copy results
        auto h_Tavg = Kokkos::create_mirror_view(Tavg);
        Kokkos::deep_copy(h_Tavg, Tavg);

        // Get interface temperatures from solver
        auto T_interface = solver->get_interface_temperatures();
        auto h_T_interface = Kokkos::create_mirror_view(T_interface);
        Kokkos::deep_copy(h_T_interface, T_interface);

        // Store average temperatures
        for (size_t j = 0; j < h_Tavg.extent(0); ++j) {
            results.average_temperatures[i * nodes.size() + j] = h_Tavg(j);
        }

        // Store interface temperatures for this pin directly from solver
        results.interface_temperatures[i].resize(h_T_interface.extent(0));
        for (size_t j = 0; j < h_T_interface.extent(0); ++j) {
            results.interface_temperatures[i][j] = h_T_interface(j);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    results.elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return results;
}

// Type aliases for specific execution spaces
using ConductionSerial = Conduction<Kokkos::Serial>;

#ifdef KOKKOS_ENABLE_OPENMP
using ConductionOpenMP = Conduction<Kokkos::OpenMP>;
#endif

#ifdef KOKKOS_ENABLE_CUDA
using ConductionCuda = Conduction<Kokkos::Cuda>;
#endif