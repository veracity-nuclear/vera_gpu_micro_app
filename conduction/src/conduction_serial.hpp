#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <Kokkos_Core.hpp>
#include "cylinder_node.hpp"
#include "cylindrical_solver_serial.hpp"
#include "materials.hpp"
#include "hdf5_utils.hpp"

class ConductionSerial {
public:
    using DoubleView = Kokkos::View<double*, Kokkos::HostSpace>;
    using DoubleView2D = Kokkos::View<double**, Kokkos::HostSpace>;

    struct ConductionResults {
        std::vector<double> average_temperatures;
        std::vector<double> interface_temperatures;
        double elapsed_time_ms;
        size_t total_pins;
    };

    ConductionSerial(
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
