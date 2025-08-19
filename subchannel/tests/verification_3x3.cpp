#include "../src/Solver.hpp"
#include <iostream>
#include <iomanip>

using namespace ants::subchannel;

/**
 * @brief 3x3 pin-centered subchannel verification case
 * 
 * This test case represents a 3x3 pin arrangement with pin-centered 
 * subchannel analysis. Each pin is centered in its own subchannel,
 * connected by gap surfaces that allow crossflow between adjacent
 * subchannels.
 * 
 * Layout (subchannel numbering):
 * [0] [1] [2]
 * [3] [4] [5]  
 * [6] [7] [8]
 * 
 * Surfaces connect adjacent subchannels:
 * - Horizontal surfaces: 0-1, 1-2, 3-4, 4-5, 6-7, 7-8
 * - Vertical surfaces: 0-3, 1-4, 2-5, 3-6, 4-7, 5-8
 */
int main() {
    try {
        std::cout << "ANTS: Alternative Nonlinear Two-phase Subchannel solver" << std::endl;
        std::cout << "3x3 Pin-Centered Subchannel Verification Case" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Create solver instance
        Solver solver;

        // Problem setup - 3x3 pin arrangement
        int n_subchannels = 9;    // 9 pin-centered subchannels
        int n_surfaces = 12;      // 12 connecting surfaces (6 horizontal + 6 vertical)
        int n_axial_nodes = 20;   // 20 axial nodes for good resolution

        // Initialize solver
        solver.initialize(n_subchannels, n_surfaces, n_axial_nodes);

        // Geometry parameters (converted from specification)
        double flow_area_m2 = 1.436e-4;        // 1.436 cm² -> m² per subchannel
        double hydraulic_diameter_m = 0.01486;   // 1.486 cm -> m
        double heated_perimeter_m = 0.01486;     // Assume same as hydraulic diameter for circular approximation
        double wetted_perimeter_m = 0.01486;     // Same as heated for single-phase approximation
        double height_m = 3.81;                  // 381.0 cm -> m
        double gap_width_m = 0.0039;            // 0.39 cm -> m
        double linear_heat_rate_W_m = 29100.0;  // 29.1 kW/m -> W/m per pin

        // Set global geometry (height, gap width)
        solver.setGeometry(flow_area_m2, heated_perimeter_m, hydraulic_diameter_m, height_m, gap_width_m);

        // Set individual subchannel geometry (all identical for this case)
        double total_flow_rate = 2.25;  // kg/s total
        double flow_per_channel = total_flow_rate / 9.0;  // kg/s per subchannel
        
        for (int i = 0; i < n_subchannels; ++i) {
            solver.setSubchannelGeometry(i, flow_area_m2, heated_perimeter_m, wetted_perimeter_m, flow_per_channel);
            solver.setSubchannelHeatRate(i, linear_heat_rate_W_m);
        }

        // Set up surface connections for 3x3 layout
        int surface_id = 0;
        
        // Horizontal surfaces (left-right connections)
        // Row 1: 0-1, 1-2
        solver.setSurfaceConnection(surface_id++, 0, 1, gap_width_m);
        solver.setSurfaceConnection(surface_id++, 1, 2, gap_width_m);
        // Row 2: 3-4, 4-5
        solver.setSurfaceConnection(surface_id++, 3, 4, gap_width_m);
        solver.setSurfaceConnection(surface_id++, 4, 5, gap_width_m);
        // Row 3: 6-7, 7-8
        solver.setSurfaceConnection(surface_id++, 6, 7, gap_width_m);
        solver.setSurfaceConnection(surface_id++, 7, 8, gap_width_m);
        
        // Vertical surfaces (top-bottom connections)
        // Col 1: 0-3, 3-6
        solver.setSurfaceConnection(surface_id++, 0, 3, gap_width_m);
        solver.setSurfaceConnection(surface_id++, 3, 6, gap_width_m);
        // Col 2: 1-4, 4-7
        solver.setSurfaceConnection(surface_id++, 1, 4, gap_width_m);
        solver.setSurfaceConnection(surface_id++, 4, 7, gap_width_m);
        // Col 3: 2-5, 5-8
        solver.setSurfaceConnection(surface_id++, 2, 5, gap_width_m);
        solver.setSurfaceConnection(surface_id++, 5, 8, gap_width_m);

        // Set operating conditions (converted from specification)
        double inlet_temperature_K = 278.0;     // 278.0 K (specified)
        double inlet_pressure_Pa = 7.255e6;     // 7.255 MPa -> Pa
        double mass_flow_rate_kg_s = 2.25;      // 2.25 kg/s total

        solver.setOperatingConditions(inlet_temperature_K, inlet_pressure_Pa, 
                                    mass_flow_rate_kg_s, linear_heat_rate_W_m);

        // Set numerical parameters for convergence
        int max_outer_iter = 50;
        int max_inner_iter = 50;
        double outer_tol = 1.0e-6;
        double inner_tol = 1.0e-14;
        double relaxation_factor = 0.5;

        solver.setNumericalParameters(max_outer_iter, max_inner_iter, outer_tol, inner_tol, relaxation_factor);

        // Print input summary
        std::cout << "\nInput Parameters:" << std::endl;
        std::cout << "  Subchannels: " << n_subchannels << std::endl;
        std::cout << "  Surfaces: " << n_surfaces << std::endl;
        std::cout << "  Flow area per channel [cm²]: " << flow_area_m2 * 1e4 << std::endl;
        std::cout << "  Hydraulic diameter [cm]: " << hydraulic_diameter_m * 1e2 << std::endl;
        std::cout << "  Height [cm]: " << height_m * 1e2 << std::endl;
        std::cout << "  Gap width [cm]: " << gap_width_m * 1e2 << std::endl;
        std::cout << "  Linear heat rate per pin [kW/m]: " << linear_heat_rate_W_m / 1000.0 << std::endl;
        std::cout << "  Inlet temperature [°C]: " << inlet_temperature_K - 273.15 << std::endl;
        std::cout << "  Inlet pressure [MPa]: " << inlet_pressure_Pa / 1e6 << std::endl;
        std::cout << "  Total flow rate [kg/s]: " << mass_flow_rate_kg_s << std::endl;
        std::cout << "  Flow rate per channel [kg/s]: " << flow_per_channel << std::endl;
        std::cout << "  Gap loss Kns: 0.5" << std::endl;

        // Solve the problem
        std::cout << "\n" << std::string(60, '-') << std::endl;
        solver.solve();

        // Print detailed results
        std::cout << "\n" << std::string(60, '-') << std::endl;
        solver.printResults();

        // Print final summary in CSV format as requested
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "FINAL RESULTS (as requested):" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Total/average bundle pressure drop [Pa]: " << solver.getBundleAveragePressureDrop() << std::endl;
        std::cout << "Exit void fraction (bundle-average) [-]: " << solver.getBundleAverageExitVoid() << std::endl;
        std::cout << "Exit temperature [°C]: " << solver.getBundleAverageExitTemperature() << std::endl;
        std::cout << "Exit temperature [K]: " << solver.getBundleAverageExitTemperatureK() << std::endl;

        // Write results to file
        solver.writeResultsToFile("ants_3x3_results.txt");
        std::cout << "\nResults written to: ants_3x3_results.txt" << std::endl;

        std::cout << "\n3x3 verification case completed successfully." << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}
