#include "Solver.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <numeric>

namespace ants {
namespace subchannel {

Solver::Solver()
    : initialized_(false)
{
    // Create the main data structures
    data_ = std::make_shared<SubchannelData>();
    properties_ = std::make_shared<Properties>();

    // Share the Properties object with SubchannelData
    data_->setProperties(properties_);
}

void Solver::initialize(int n_subchannels, int n_surfaces, int n_axial_nodes) {
    if (n_subchannels <= 0 || n_axial_nodes <= 0) {
        throw std::invalid_argument("Invalid number of subchannels or axial nodes");
    }

    if (n_surfaces < 0) {
        throw std::invalid_argument("Invalid number of surfaces");
    }

    // Initialize the data structure
    data_->initialize(n_subchannels, n_surfaces, n_axial_nodes);

    // Create thermal hydraulics solver
    th_solver_ = std::make_shared<ThermalHydraulics>(data_, properties_);

    initialized_ = true;
}

void Solver::setGeometry(double flow_area, double heated_perimeter,
                        double hydraulic_diameter, double axial_height, double gap_width) {
    if (!initialized_) {
        throw std::runtime_error("Solver must be initialized before setting geometry");
    }

    data_->flow_area = flow_area;
    data_->heated_perimeter = heated_perimeter;
    data_->wetted_perimeter = heated_perimeter; // Assume same for now
    data_->hydraulic_diameter = hydraulic_diameter;
    data_->axial_height = axial_height;
    data_->gap_width = gap_width;
    data_->node_height = axial_height / data_->getNumAxialNodes();
}

void Solver::setOperatingConditions(double inlet_temp, double inlet_pressure,
                                   double mass_flow_rate, double linear_heat_rate) {
    if (!initialized_) {
        throw std::runtime_error("Solver must be initialized before setting operating conditions");
    }

    data_->inlet_temperature = inlet_temp;
    data_->inlet_pressure = inlet_pressure;
    data_->mass_flow_rate = mass_flow_rate;
    data_->linear_heat_rate = linear_heat_rate;

    // Update properties and derived quantities
    properties_->setOperatingPressure(inlet_pressure);
    data_->inlet_enthalpy = properties_->liquidEnthalpy(inlet_temp, inlet_pressure);

    // Update heat flux
    double heat_flux = linear_heat_rate / data_->heated_perimeter;
    std::fill(data_->heat_flux.begin(), data_->heat_flux.end(), heat_flux);

    // Finalize initialization with proper property values
    data_->finalizeInitialization();
}

void Solver::setNumericalParameters(int max_outer_iter, int max_inner_iter,
                                   double outer_tol, double inner_tol, double relax_factor) {
    if (!initialized_) {
        throw std::runtime_error("Solver must be initialized before setting numerical parameters");
    }

    data_->max_outer_iterations = max_outer_iter;
    data_->max_inner_iterations = max_inner_iter;
    data_->outer_tolerance = outer_tol;
    data_->inner_tolerance = inner_tol;
    data_->relaxation_factor = relax_factor;
}

void Solver::setSubchannelGeometry(int subchannel_id, double flow_area, double heated_perimeter,
                                  double wetted_perimeter, double mass_flow_rate) {
    if (!initialized_) {
        throw std::runtime_error("Solver must be initialized before setting subchannel geometry");
    }

    if (subchannel_id < 0 || subchannel_id >= data_->getNumSubchannels()) {
        throw std::invalid_argument("Invalid subchannel ID");
    }

    // For now, store in global variables (could be extended to per-subchannel arrays)
    // This is a simplified implementation - full version would have per-subchannel properties
    if (subchannel_id == 0) {
        data_->flow_area = flow_area;
        data_->heated_perimeter = heated_perimeter;
        data_->wetted_perimeter = wetted_perimeter;
        if (mass_flow_rate > 0.0) {
            data_->mass_flow_rate = mass_flow_rate;
        }
    }
}

void Solver::setSubchannelHeatRate(int subchannel_id, double linear_heat_rate) {
    if (!initialized_) {
        throw std::runtime_error("Solver must be initialized before setting subchannel heat rate");
    }

    if (subchannel_id < 0 || subchannel_id >= data_->getNumSubchannels()) {
        throw std::invalid_argument("Invalid subchannel ID");
    }

    // Calculate heat flux for this subchannel
    double heat_flux = linear_heat_rate / data_->heated_perimeter;
    
    // Set heat flux for all axial cells in this subchannel
    int n_axial_nodes = data_->getNumAxialNodes();
    for (int k = 0; k < n_axial_nodes; ++k) {
        int cell_index = subchannel_id * n_axial_nodes + k;
        data_->heat_flux[cell_index] = heat_flux;
    }
    
    // Debug output
    std::cout << "Set subchannel " << subchannel_id << " heat rate: " << linear_heat_rate 
              << " W/m -> heat_flux = " << heat_flux << " W/m² (cells " 
              << subchannel_id * n_axial_nodes << "-" << (subchannel_id + 1) * n_axial_nodes - 1 << ")" << std::endl;
}

void Solver::setSurfaceConnection(int surface_id, int subchannel_in, int subchannel_out,
                                 double gap_width, double loss_coefficient) {
    if (!initialized_) {
        throw std::runtime_error("Solver must be initialized before setting surface connections");
    }

    if (surface_id < 0 || surface_id >= data_->getNumSurfaces()) {
        throw std::invalid_argument("Invalid surface ID");
    }

    if (subchannel_in < 0 || subchannel_in >= data_->getNumSubchannels() ||
        subchannel_out < 0 || subchannel_out >= data_->getNumSubchannels()) {
        throw std::invalid_argument("Invalid subchannel IDs for surface connection");
    }

    // Initialize surface connectivity if not already done
    if (data_->surface_connections.size() != static_cast<size_t>(data_->getNumSurfaces())) {
        data_->surface_connections.resize(data_->getNumSurfaces());
        data_->subchannel_surfaces.resize(data_->getNumSubchannels());
    }

    // Set up surface connection: [in_subchannel, out_subchannel]
    data_->surface_connections[surface_id] = {subchannel_in, subchannel_out};

    // Add surface to both subchannels' surface lists
    data_->subchannel_surfaces[subchannel_in].push_back(surface_id);
    data_->subchannel_surfaces[subchannel_out].push_back(surface_id);

    // Store gap width and loss coefficient (simplified storage for now)
    data_->gap_width = gap_width;
    data_->loss_coefficient = loss_coefficient;
}

void Solver::solve() {
    solveSteadyState();
}

void Solver::solveSteadyState() {
    if (!initialized_) {
        throw std::runtime_error("Solver not initialized");
    }

    validateConfiguration();

    std::cout << "Starting ANTS subchannel solution..." << std::endl;
    std::cout << "Subchannels: " << data_->getNumSubchannels() << std::endl;
    std::cout << "Axial nodes: " << data_->getNumAxialNodes() << std::endl;
    std::cout << "Surfaces: " << data_->getNumSurfaces() << std::endl;

    // Solve the thermal hydraulics
    th_solver_->solve();

    std::cout << "Solution completed." << std::endl;
}

void Solver::printResults() const {
    if (!initialized_) {
        std::cout << "Solver not initialized" << std::endl;
        return;
    }

    th_solver_->printSolution();
}

void Solver::printSummary() const {
    if (!initialized_) {
        std::cout << "Solver not initialized" << std::endl;
        return;
    }

    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "ANTS: Alternative Nonlinear Two-phase Subchannel solver" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total/average bundle pressure drop [Pa]: " << getBundleAveragePressureDrop() << std::endl;
    std::cout << "Exit void fraction (bundle-average) [-]: " << getBundleAverageExitVoid() << std::endl;
    std::cout << "Exit temperature [°C]: " << getBundleAverageExitTemperature() << std::endl;
    std::cout << "Exit temperature [K]: " << getBundleAverageExitTemperatureK() << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void Solver::writeResultsToFile(const std::string& filename) const {
    if (!initialized_) {
        throw std::runtime_error("Solver not initialized");
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    file << "ANTS Subchannel Results" << std::endl;
    file << "======================" << std::endl;
    file << std::fixed << std::setprecision(6);

    file << "Geometry:" << std::endl;
    file << "  Flow area [cm²]: " << data_->flow_area * 1e4 << std::endl;
    file << "  Hydraulic diameter [cm]: " << data_->hydraulic_diameter * 1e2 << std::endl;
    file << "  Height [cm]: " << data_->axial_height * 1e2 << std::endl;
    file << "  Gap width [cm]: " << data_->gap_width * 1e2 << std::endl;

    file << "\nOperating Conditions:" << std::endl;
    file << "  Linear heat rate [kW/m]: " << data_->linear_heat_rate / 1000.0 << std::endl;
    file << "  Inlet temperature [°C]: " << data_->inlet_temperature - 273.15 << std::endl;
    file << "  Inlet pressure [MPa]: " << data_->inlet_pressure / 1e6 << std::endl;
    file << "  Inlet flow rate [kg/s]: " << data_->mass_flow_rate << std::endl;
    file << "  Gap loss Kns: " << data_->loss_coefficient << std::endl;

    file << "\nResults Summary:" << std::endl;
    file << "  Total/average bundle pressure drop [Pa]: " << getBundleAveragePressureDrop() << std::endl;
    file << "  Exit void fraction (bundle-average) [-]: " << getBundleAverageExitVoid() << std::endl;
    file << "  Exit temperature [°C]: " << getBundleAverageExitTemperature() << std::endl;
    file << "  Exit temperature [K]: " << getBundleAverageExitTemperatureK() << std::endl;

    file << "\nDetailed Results:" << std::endl;
    for (int ch = 0; ch < data_->getNumSubchannels(); ++ch) {
        file << "\nChannel " << ch << ":" << std::endl;
        file << "Axial Node | Void [-] | Quality [-] | Pressure [Pa] | Temperature [K]" << std::endl;
        file << std::string(70, '-') << std::endl;

        for (int k = 0; k <= data_->getNumAxialNodes(); ++k) {
            int idx = ch * (data_->getNumAxialNodes() + 1) + k;
            double temp = properties_->liquidTemperature(data_->pressure[idx],
                                                       data_->liquid_enthalpy[idx]);

            file << std::setw(10) << k
                 << " | " << std::setw(8) << data_->void_fraction[idx]
                 << " | " << std::setw(11) << data_->flow_quality[idx]
                 << " | " << std::setw(13) << data_->pressure[idx]
                 << " | " << std::setw(14) << temp << std::endl;
        }
    }

    file.close();
}

double Solver::getExitVoidFraction(int channel_id) const {
    return th_solver_->getExitVoidFraction(channel_id);
}

double Solver::getExitTemperature(int channel_id) const {
    return th_solver_->getExitTemperature(channel_id);
}

double Solver::getExitTemperatureK(int channel_id) const {
    return th_solver_->getExitTemperatureK(channel_id);
}

double Solver::getPressureDrop(int channel_id) const {
    return th_solver_->getPressureDrop(channel_id);
}

double Solver::getBundleAveragePressureDrop() const {
    if (!initialized_) {
        return 0.0;
    }

    // For now, just return channel 0 result
    // In multichannel case, would calculate flow-weighted average
    return getPressureDrop(0);
}

double Solver::getBundleAverageExitVoid() const {
    if (!initialized_) {
        return 0.0;
    }

    // For now, just return channel 0 result
    // In multichannel case, would calculate flow-weighted average
    return getExitVoidFraction(0);
}

double Solver::getBundleAverageExitTemperature() const {
    if (!initialized_) {
        return 0.0;
    }

    // For now, just return channel 0 result
    // In multichannel case, would calculate flow-weighted average
    return getExitTemperature(0);
}

double Solver::getBundleAverageExitTemperatureK() const {
    if (!initialized_) {
        return 0.0;
    }

    // For now, just return channel 0 result
    // In multichannel case, would calculate flow-weighted average
    return getExitTemperatureK(0);
}

void Solver::validateConfiguration() const {
    if (data_->flow_area <= 0.0) {
        throw std::runtime_error("Flow area must be positive");
    }

    if (data_->heated_perimeter <= 0.0) {
        throw std::runtime_error("Heated perimeter must be positive");
    }

    if (data_->axial_height <= 0.0) {
        throw std::runtime_error("Axial height must be positive");
    }

    if (data_->mass_flow_rate <= 0.0) {
        throw std::runtime_error("Mass flow rate must be positive");
    }

    if (data_->inlet_pressure <= 0.0) {
        throw std::runtime_error("Inlet pressure must be positive");
    }

    if (data_->inlet_temperature <= 273.15) {
        throw std::runtime_error("Inlet temperature must be above freezing");
    }
}

} // namespace subchannel
} // namespace ants
