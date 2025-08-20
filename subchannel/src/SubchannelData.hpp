#ifndef ANTS_SUBCHANNEL_SUBCHANNEL_DATA_HPP
#define ANTS_SUBCHANNEL_SUBCHANNEL_DATA_HPP

#include "Constants.hpp"
#include <vector>
#include <memory>

namespace ants {
namespace subchannel {

class Properties; // Forward declaration

/**
 * @brief Data structures for subchannel thermal hydraulic analysis
 *
 * This class contains the data structures and variables needed for
 * subchannel analysis, corresponding to the relevant portions of FLUX_MODB.
 */
class SubchannelData {
public:
    SubchannelData();
    ~SubchannelData() = default;

    // Initialize data structures
    void initialize(int n_subchannels, int n_surfaces, int n_axial_nodes);
    void clear();

    // Set properties object
    void setProperties(std::shared_ptr<Properties> props);

    // Finalize initialization after properties are set
    void finalizeInitialization();

    // Geometry parameters
    double flow_area;           // Flow area [m^2]
    double heated_perimeter;    // Heated perimeter [m]
    double wetted_perimeter;    // Wetted perimeter [m]
    double hydraulic_diameter;  // Hydraulic diameter [m]
    double axial_height;        // Total axial height [m]
    double node_height;         // Axial node height [m]
    double gap_width;           // Gap width [m]
    double gap_spacing;         // Transverse path length [m]

    // Operating conditions
    double inlet_temperature;   // Inlet temperature [K]
    double inlet_pressure;      // Inlet pressure [Pa]
    double inlet_enthalpy;      // Inlet enthalpy [J/kg]
    double mass_flow_rate;      // Mass flow rate [kg/s]
    double linear_heat_rate;    // Linear heat rate [W/m]
    double power_scaling;       // Power scaling factor

    // Numerical parameters
    int max_outer_iterations;   // Maximum outer iterations
    int max_inner_iterations;   // Maximum inner iterations
    double outer_tolerance;     // Outer convergence tolerance
    double inner_tolerance;     // Inner convergence tolerance
    double relaxation_factor;   // Relaxation factor for crossflow

    // Two-phase flow parameters
    double mixing_parameter;    // Turbulent mixing parameter
    double void_drift_parameter; // Void drift parameter
    double aspect_ratio;        // Subchannel aspect ratio
    double loss_coefficient;    // Form loss coefficient

    // Solution arrays - current values (axial edge values, 0:n_axial_nodes)
    std::vector<double> liquid_mass_flow;    // Liquid mass flow rate [kg/s]
    std::vector<double> vapor_mass_flow;     // Vapor mass flow rate [kg/s]
    std::vector<double> total_mass_flow;     // Total mass flow rate [kg/s]
    std::vector<double> liquid_enthalpy;     // Liquid enthalpy [J/kg]
    std::vector<double> mixture_enthalpy;    // Mixture enthalpy [J/kg]
    std::vector<double> liquid_density;      // Liquid density [kg/m^3]
    std::vector<double> flow_quality;        // Flow quality [-]
    std::vector<double> void_fraction;       // Void fraction [-]
    std::vector<double> pressure;            // Pressure [Pa]
    std::vector<double> mixture_velocity;    // Mixture velocity [m/s]
    std::vector<double> momentum_flux;       // G*V momentum flux [kg/m-s^2]

    // Cell-centered arrays (1:n_axial_nodes)
    std::vector<double> heat_flux;           // Heat flux [W/m^2]
    std::vector<double> evaporation_rate;    // Evaporation mass flux [kg/m^2-s]

    // Crossflow arrays for multichannel analysis
    std::vector<double> crossflow_mass_flux;     // Surface crossflow mass flux [kg/m^2-s]
    std::vector<double> crossflow_momentum;      // Surface crossflow momentum [kg/m-s^2]

    // Surface connectivity for multichannel problems
    std::vector<std::vector<int>> surface_neighbors;   // Neighboring subchannel indices
    std::vector<std::vector<int>> subchannel_surfaces; // Surface indices for each subchannel
    std::vector<std::vector<int>> surface_connections; // Surface connection topology

    // Boundary condition arrays
    std::vector<int> surface_bc_types;       // Boundary condition types

    // Property objects and working arrays
    std::shared_ptr<Properties> properties;

    // Axial mesh
    std::vector<double> axial_mesh;          // Axial mesh positions [m]

    // Getters for key parameters
    int getNumSubchannels() const { return n_subchannels_; }
    int getNumSurfaces() const { return n_surfaces_; }
    int getNumAxialNodes() const { return n_axial_nodes_; }
    double getNodeHeight() const { return node_height; }

    // Solution state access
    const std::vector<double>& getLiquidMassFlow() const { return liquid_mass_flow; }
    const std::vector<double>& getVaporMassFlow() const { return vapor_mass_flow; }
    const std::vector<double>& getVoidFraction() const { return void_fraction; }
    const std::vector<double>& getPressure() const { return pressure; }
    const std::vector<double>& getLiquidEnthalpy() const { return liquid_enthalpy; }

private:
    int n_subchannels_;
    int n_surfaces_;
    int n_axial_nodes_;

    // Helper functions
    void allocateArrays();
    void setDefaultValues();
};

} // namespace subchannel
} // namespace ants

#endif // ANTS_SUBCHANNEL_SUBCHANNEL_DATA_HPP
