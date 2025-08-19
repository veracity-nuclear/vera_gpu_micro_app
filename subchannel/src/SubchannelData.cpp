#include "SubchannelData.hpp"
#include "Properties.hpp"
#include <stdexcept>

namespace ants {
namespace subchannel {

SubchannelData::SubchannelData() 
    : flow_area(1.436e-4)           // 1.436 cm^2 -> m^2
    , heated_perimeter(0.01486)     // 1.486 cm -> m  
    , wetted_perimeter(0.01486)
    , hydraulic_diameter(0.01486)   // 1.486 cm -> m
    , axial_height(3.81)            // 381 cm -> m
    , node_height(0.1)              // Default 10 cm nodes
    , gap_width(0.0039)             // 0.39 cm -> m
    , inlet_temperature(278.0)      // K
    , inlet_pressure(7.255e6)       // Pa
    , inlet_enthalpy(0.0)
    , mass_flow_rate(2.25)          // kg/s
    , linear_heat_rate(29100.0)     // 29.1 kW/m -> W/m
    , power_scaling(1.0)
    , max_outer_iterations(50)
    , max_inner_iterations(50)
    , outer_tolerance(1.0e-6)
    , inner_tolerance(1.0e-14)
    , relaxation_factor(0.5)
    , mixing_parameter(0.01)
    , void_drift_parameter(1.0)
    , aspect_ratio(1.0)
    , loss_coefficient(0.5)
    , n_subchannels_(0)
    , n_surfaces_(0)
    , n_axial_nodes_(0)
{
    // Properties object will be set via setProperties()
}

void SubchannelData::setProperties(std::shared_ptr<Properties> props) {
    properties = props;
}

void SubchannelData::finalizeInitialization() {
    if (!properties) {
        throw std::runtime_error("Properties object not set before finalizing initialization");
    }
    
    // Now that properties are set, update property-dependent values
    double rho_liquid = properties->getLiquidDensity();
    std::fill(liquid_density.begin(), liquid_density.end(), rho_liquid);
    
    double velocity = mass_flow_rate / (flow_area * rho_liquid);
    std::fill(mixture_velocity.begin(), mixture_velocity.end(), velocity);
}

void SubchannelData::initialize(int n_subchannels, int n_surfaces, int n_axial_nodes) {
    n_subchannels_ = n_subchannels;
    n_surfaces_ = n_surfaces;
    n_axial_nodes_ = n_axial_nodes;
    
    // Calculate derived parameters
    node_height = axial_height / n_axial_nodes;
    hydraulic_diameter = 4.0 * flow_area / wetted_perimeter;
    
    // Initialize properties at operating conditions (only if properties object is set)
    if (properties) {
        properties->setOperatingPressure(inlet_pressure);
        inlet_enthalpy = properties->enthalpyFromTemperature(inlet_temperature);
    }
    
    allocateArrays();
    setDefaultValues();
    
    // Build axial mesh
    axial_mesh.resize(n_axial_nodes + 1);
    for (int k = 0; k <= n_axial_nodes; ++k) {
        axial_mesh[k] = k * node_height;
    }
}

void SubchannelData::allocateArrays() {
    // Axial arrays sized (n_subchannels * (n_axial_nodes + 1))
    int axial_size = n_subchannels_ * (n_axial_nodes_ + 1);
    
    liquid_mass_flow.resize(axial_size);
    vapor_mass_flow.resize(axial_size);
    total_mass_flow.resize(axial_size);
    liquid_enthalpy.resize(axial_size);
    mixture_enthalpy.resize(axial_size);
    liquid_density.resize(axial_size);
    flow_quality.resize(axial_size);
    void_fraction.resize(axial_size);
    pressure.resize(axial_size);
    mixture_velocity.resize(axial_size);
    momentum_flux.resize(axial_size);
    
    // Cell-centered arrays sized (n_subchannels * n_axial_nodes)
    int cell_size = n_subchannels_ * n_axial_nodes_;
    
    heat_flux.resize(cell_size);
    evaporation_rate.resize(cell_size);
    
    // Surface arrays
    if (n_surfaces_ > 0) {
        int surface_axial_size = n_surfaces_ * (n_axial_nodes_ + 1);
        crossflow_mass_flux.resize(surface_axial_size);
        crossflow_momentum.resize(surface_axial_size);
        
        surface_bc_types.resize(n_surfaces_);
        
        // Surface connectivity - resize to accommodate topology
        surface_neighbors.resize(n_surfaces_);
        subchannel_surfaces.resize(n_subchannels_);
        surface_connections.resize(n_surfaces_);
    }
}

void SubchannelData::setDefaultValues() {
    // Initialize with inlet conditions - but avoid property calls if properties not ready
    std::fill(liquid_mass_flow.begin(), liquid_mass_flow.end(), mass_flow_rate);
    std::fill(vapor_mass_flow.begin(), vapor_mass_flow.end(), 0.0);
    std::fill(total_mass_flow.begin(), total_mass_flow.end(), mass_flow_rate);
    std::fill(liquid_enthalpy.begin(), liquid_enthalpy.end(), inlet_enthalpy);
    std::fill(mixture_enthalpy.begin(), mixture_enthalpy.end(), inlet_enthalpy);
    
    // Only set property-dependent values if properties object is available and initialized
    if (properties && inlet_pressure > 0.0) {
        std::fill(liquid_density.begin(), liquid_density.end(), properties->getLiquidDensity());
        std::fill(mixture_velocity.begin(), mixture_velocity.end(), 
                  mass_flow_rate / (flow_area * properties->getLiquidDensity()));
    } else {
        // Use placeholder values
        std::fill(liquid_density.begin(), liquid_density.end(), 750.0);  // kg/m³
        std::fill(mixture_velocity.begin(), mixture_velocity.end(), 1.0);   // m/s
    }
    
    std::fill(flow_quality.begin(), flow_quality.end(), 0.0);
    std::fill(void_fraction.begin(), void_fraction.end(), 0.0);
    std::fill(pressure.begin(), pressure.end(), inlet_pressure);
    std::fill(momentum_flux.begin(), momentum_flux.end(), 0.0);
    
    // Initialize heat flux
    double uniform_heat_flux = linear_heat_rate / heated_perimeter;
    std::fill(heat_flux.begin(), heat_flux.end(), uniform_heat_flux);
    std::fill(evaporation_rate.begin(), evaporation_rate.end(), 0.0);
    
    // Initialize crossflow arrays
    if (n_surfaces_ > 0) {
        std::fill(crossflow_mass_flux.begin(), crossflow_mass_flux.end(), 0.0);
        std::fill(crossflow_momentum.begin(), crossflow_momentum.end(), 0.0);
        std::fill(surface_bc_types.begin(), surface_bc_types.end(), 0);
    }
}

void SubchannelData::clear() {
    liquid_mass_flow.clear();
    vapor_mass_flow.clear(); 
    total_mass_flow.clear();
    liquid_enthalpy.clear();
    mixture_enthalpy.clear();
    liquid_density.clear();
    flow_quality.clear();
    void_fraction.clear();
    pressure.clear();
    mixture_velocity.clear();
    momentum_flux.clear();
    
    heat_flux.clear();
    evaporation_rate.clear();
    
    crossflow_mass_flux.clear();
    crossflow_momentum.clear();
    
    surface_neighbors.clear();
    subchannel_surfaces.clear();
    surface_connections.clear();
    surface_bc_types.clear();
    
    axial_mesh.clear();
    
    n_subchannels_ = 0;
    n_surfaces_ = 0;
    n_axial_nodes_ = 0;
}

} // namespace subchannel
} // namespace ants
