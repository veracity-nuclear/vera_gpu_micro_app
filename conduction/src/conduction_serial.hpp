#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <numeric>
#include <iostream>
#include <cassert>
#include <exception>
#include <mutex>
#include <functional>
#include <Kokkos_Core.hpp>
#include "cylinder_node.hpp"
#include "cylindrical_solver_serial.hpp"
#include "materials.hpp"
#include <highfive/H5File.hpp>

template<typename ExecutionSpace>
class Conduction {
public:
    using execution_space = ExecutionSpace;
    using memory_space = typename execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    using DoubleView1D = Kokkos::View<double*, device_type>;
    using DoubleView2D = Kokkos::View<double**, device_type>;

    struct ConductionResults {
        std::vector<double> average_temperatures;                 // [pin * num_nodes + node]
        std::vector<std::vector<double>> interface_temperatures;  // per pin, inner to outer: fuel CL, fuel surface, clad inner, clad outer
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
    if (radii_.size() != 4 || materials_.size() != 2) {  // (For now, can expand later)
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

    // Open the file once for all datasets (the generic helpers in hdf5_utils.hpp reopen it per read)
    HighFive::File file(filename, HighFive::File::ReadOnly);

    auto read_scalar = [&file](const std::string& path) {
        double value;
        file.getDataSet(path).read(value);
        return value;
    };

    // Read a dataset of any shape, flattened, appending it to `out`; returns the number of values read
    auto append_flat = [&file](const std::string& path, std::vector<double>& out) {
        HighFive::DataSet dataset = file.getDataSet(path);
        std::vector<size_t> dims = dataset.getSpace().getDimensions();
        size_t count = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
        size_t offset = out.size();
        out.resize(offset + count);
        dataset.read_raw(out.data() + offset);
        return count;
    };

    for (size_t i = 0; i < state_groups.size(); ++i) {
        const auto& group = state_groups[i];

        double total_power = read_scalar(group + "/total_power");
        double power_percent = read_scalar(group + "/power") * 0.01; // convert to fraction

        size_t offset = pin_powers_.size();
        append_flat(group + "/pin_powers", pin_powers_);
        auto first = pin_powers_.begin() + offset;
        double scaling = total_power * power_percent / std::accumulate(first, pin_powers_.end(), 0.0);
        for (auto it = first; it != pin_powers_.end(); ++it) *it *= scaling;

        offset = clad_surf_temps_.size();
        append_flat(group + "/pin_max_clad_surface_temp", clad_surf_temps_);
        for (auto it = clad_surf_temps_.begin() + offset; it != clad_surf_temps_.end(); ++it) *it += 273.15;  // convert to K
    }

    data_loaded_ = true;
}

template<typename ExecutionSpace>
typename Conduction<ExecutionSpace>::ConductionResults
Conduction<ExecutionSpace>::solve_all_pins() {
    if (!data_loaded_) {
        throw std::runtime_error("Pin data not loaded. Call load_pin_data() first.");
    }

    // The per-pin solve calls virtual material methods and may throw, so it can only run on an
    // execution space that can access host memory (Serial, OpenMP, ...).
    static_assert(Kokkos::SpaceAccessibility<execution_space, Kokkos::HostSpace>::accessible,
                  "Conduction::solve_all_pins requires a host-accessible execution space");

    assert(pin_powers_.size() == clad_surf_temps_.size());
    const size_t N = pin_powers_.size();

    // Every pin has the same as-fabricated geometry: build the prototype nodes, validate them and
    // compute the interface layout once, instead of once per pin.
    constexpr size_t num_nodes = 2; // fuel, clad (constructor enforces 4 radii / 2 materials)
    constexpr size_t max_interfaces = 2 * num_nodes;
    constexpr double T_initial = 600.0; // K, same default as CylindricalSolverSerial
    constexpr double tolerance = 1e-6;
    constexpr size_t max_iterations = 100;

    CylinderNode prototype[num_nodes] = {
        CylinderNode(height_, radii_[0], radii_[1]), // fuel node
        CylinderNode(height_, radii_[2], radii_[3])  // clad node
    };
    for (auto& node : prototype) node.set_temperature(T_initial);
    cylindrical_solver_detail::validate_radii(prototype, num_nodes);

    std::pair<size_t, size_t> node_to_interface[num_nodes];
    const size_t num_interfaces = cylindrical_solver_detail::build_interface_layout(prototype, num_nodes, node_to_interface);
    const double fuel_volume = prototype[0].get_volume();

    ConductionResults results;
    results.total_pins = N;
    results.average_temperatures.resize(N * num_nodes);
    results.interface_temperatures.resize(N);

    auto start = std::chrono::high_resolution_clock::now();

    // Host-accessible execution space: wrap existing host buffers in unmanaged Views instead of
    // allocating, initializing and deep-copying separate device Views.
    using UnmanagedConst1D = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Unmanaged2D = Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    UnmanagedConst1D pin_powers_view(pin_powers_.data(), N);
    UnmanagedConst1D clad_temps_view(clad_surf_temps_.data(), N);
    Unmanaged2D Tavg_results_view(results.average_temperatures.data(), N, num_nodes);
    Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> T_interface_results_view(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "T_interface_results"), N, num_interfaces);

    // Shared, read-only state. Materials and the gap fluid are only used through const methods,
    // so a single instance is safe to share across threads.
    const Solid* materials[num_nodes] = {materials_[0].get(), materials_[1].get()};
    const Helium gap_fluid_instance("He"); // same default gap fluid as CylindricalSolverSerial
    const Fluid* gap_fluid = &gap_fluid_instance;

    // Exceptions must not escape an OpenMP parallel region; capture the first one and rethrow it
    // after the kernel (identical behaviour to the serial path, where it would propagate).
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::exception_ptr* first_error_ptr = &first_error;
    std::mutex* error_mutex_ptr = &error_mutex;

    // Hot loop: allocation-free. Each pin's mutable state (two CylinderNode copies of the
    // prototype, interface temperatures, solver workspace) lives in fixed-size stack arrays local
    // to the iteration, and the solver core (cylindrical_solver_detail::solve) only works on that
    // caller-provided storage. So there is no heap traffic, no allocator contention between
    // threads, and no shared mutable state; outputs go to disjoint rows of pre-allocated Views.
    // (Only the error path allocates: an exception object, which is captured below.)
    Kokkos::parallel_for("Outermost Loop", Kokkos::RangePolicy<execution_space>(0, N), [=](const size_t i) {
        try {
            CylinderNode nodes[num_nodes] = {prototype[0], prototype[1]};
            const double qdot[num_nodes] = {pin_powers_view(i) / fuel_volume, 0.0}; // only fuel node has heat generation
            double interface_temps[max_interfaces];
            for (size_t j = 0; j < num_interfaces; ++j) interface_temps[j] = T_initial;

            double Tavg[num_nodes], Tavg_prev[num_nodes], inner_Tavg_prev[num_nodes], volumes[num_nodes], k[num_nodes];
            CylinderNode::LogRadii logs[num_nodes];
            cylindrical_solver_detail::Workspace ws{Tavg, Tavg_prev, inner_Tavg_prev, volumes, k, logs};

            bool converged = cylindrical_solver_detail::solve(
                nodes, materials, num_nodes, *gap_fluid, node_to_interface, num_interfaces,
                interface_temps, qdot, clad_temps_view(i), tolerance, max_iterations, ws);
            if (!converged) {
                throw std::runtime_error("Cylindrical solver did not converge");
            }

            for (size_t j = 0; j < num_nodes; ++j) {
                Tavg_results_view(i, j) = Tavg[j];
            }
            for (size_t j = 0; j < num_interfaces; ++j) {
                T_interface_results_view(i, j) = interface_temps[j];
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(*error_mutex_ptr);
            if (!*first_error_ptr) *first_error_ptr = std::current_exception();
        }
    });
    Kokkos::fence();

    if (first_error) {
        std::rethrow_exception(first_error);
    }

    // Average temperatures were written in place; unpack interface temperatures per pin
    for (size_t i = 0; i < N; ++i) {
        results.interface_temperatures[i].assign(&T_interface_results_view(i, 0), &T_interface_results_view(i, 0) + num_interfaces);
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