#include "conduction_serial.hpp"
#include <numeric>
#include <iostream>
#include <cassert>

ConductionSerial::ConductionSerial(
    const std::vector<double>& radii,
    double height,
    const std::vector<std::shared_ptr<Solid>>& materials
) : radii_(radii), height_(height), materials_(materials) {
    // For this specific use case, we expect:
    // - 4 radii: [fuel_inner, fuel_outer, clad_inner, clad_outer]
    // - 2 materials: [fuel, clad]
    // - There's a gap between fuel_outer and clad_inner
    if (radii_.size() != 4 || materials_.size() != 2) {
        throw std::invalid_argument("Expected 4 radii and 2 materials for fuel-gap-clad geometry");
    }
}

void ConductionSerial::load_pin_data(
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

ConductionSerial::ConductionResults ConductionSerial::solve_all_pins() {
    if (!data_loaded_) {
        throw std::runtime_error("Pin data not loaded. Call load_pin_data() first.");
    }

    assert(pin_powers_.size() == clad_surf_temps_.size());
    size_t N = pin_powers_.size();

    ConductionResults results;
    results.total_pins = N;
    results.average_temperatures.resize(N * 2); // 2 nodes per pin (fuel and clad)
    results.interface_temperatures.resize(N);

    auto start = std::chrono::high_resolution_clock::now();

    // Use simple for loop for now - we can optimize with parallel_for later if needed
    for (size_t i = 0; i < N; ++i) {
        // Create nodes for this pin (fuel and clad with gap between)
        std::vector<std::shared_ptr<CylinderNode>> nodes = {
            std::make_shared<CylinderNode>(height_, radii_[0], radii_[1]), // fuel node
            std::make_shared<CylinderNode>(height_, radii_[2], radii_[3])  // clad node
        };

        // Create solver for this pin
        CylindricalSolverSerial solver(nodes, materials_);
        double T_outer = clad_surf_temps_[i];

        // Create Kokkos::View for qdot
        CylindricalSolverSerial::DoubleView qdot("qdot", nodes.size());
        auto h_qdot = Kokkos::create_mirror_view(qdot);

        for (size_t j = 0; j < nodes.size(); ++j) {
            h_qdot(j) = 0.0;
        }
        h_qdot(0) = pin_powers_[i] / nodes[0]->get_volume(); // only fuel node has heat generation
        Kokkos::deep_copy(qdot, h_qdot);

        // Solve for this pin
        CylindricalSolverSerial::DoubleView Tavg = solver.solve(qdot, T_outer);

        // Copy results
        auto h_Tavg = Kokkos::create_mirror_view(Tavg);
        Kokkos::deep_copy(h_Tavg, Tavg);

        double temp_sum = 0.0;
        for (size_t j = 0; j < h_Tavg.extent(0); ++j) {
            results.average_temperatures[i * nodes.size() + j] = h_Tavg(j);
            temp_sum += h_Tavg(j);
        }
        results.interface_temperatures[i] = temp_sum; // Store sum for validation
    }

    auto end = std::chrono::high_resolution_clock::now();
    results.elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return results;
}
