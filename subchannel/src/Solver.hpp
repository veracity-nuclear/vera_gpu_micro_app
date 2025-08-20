#ifndef ANTS_SUBCHANNEL_SOLVER_HPP
#define ANTS_SUBCHANNEL_SOLVER_HPP

#include "Constants.hpp"
#include "SubchannelData.hpp"
#include "Properties.hpp"
#include "ThermalHydraulics.hpp"
#include <memory>
#include <string>

namespace ants {
namespace subchannel {

/**
 * @brief Main ANTS subchannel solver class
 *
 * This class coordinates the overall subchannel analysis,
 * managing the solution process and providing the main interface.
 */
class Solver {
public:
    Solver();
    ~Solver() = default;

    // Setup and configuration
    void initialize(int n_subchannels, int n_surfaces, int n_axial_nodes);
    void setGeometry(double flow_area, double heated_perimeter,
                    double hydraulic_diameter, double axial_height, double gap_width);
    void setOperatingConditions(double inlet_temp, double inlet_pressure,
                              double mass_flow_rate, double linear_heat_rate);
    void setNumericalParameters(int max_outer_iter, int max_inner_iter,
                               double outer_tol, double inner_tol, double relax_factor);

    // Individual subchannel and surface setup
    void setSubchannelGeometry(int subchannel_id, double flow_area, double heated_perimeter,
                              double wetted_perimeter, double mass_flow_rate = -1.0);
    void setSubchannelHeatRate(int subchannel_id, double linear_heat_rate);
    void setSurfaceConnection(int surface_id, int subchannel_in, int subchannel_out,
                            double gap_width, double loss_coefficient = 0.5);

    // Main solve interface
    void solve();
    void solveSteadyState();

    // Results and output
    void printResults() const;
    void printSummary() const;
    void writeResultsToFile(const std::string& filename) const;

    // Result accessors
    double getExitVoidFraction(int channel_id = 0) const;
    double getExitTemperature(int channel_id = 0) const;      // [°C]
    double getExitTemperatureK(int channel_id = 0) const;     // [K]
    double getPressureDrop(int channel_id = 0) const;         // [Pa]
    double getBundleAveragePressureDrop() const;              // [Pa]
    double getBundleAverageExitVoid() const;                  // [-]
    double getBundleAverageExitTemperature() const;           // [°C]
    double getBundleAverageExitTemperatureK() const;          // [K]

    // Access to internal objects for advanced use
    std::shared_ptr<SubchannelData> getData() { return data_; }
    std::shared_ptr<Properties> getProperties() { return properties_; }
    std::shared_ptr<ThermalHydraulics> getThermalHydraulics() { return th_solver_; }

private:
    std::shared_ptr<SubchannelData> data_;
    std::shared_ptr<Properties> properties_;
    std::shared_ptr<ThermalHydraulics> th_solver_;

    bool initialized_;

    // Helper functions
    void validateConfiguration() const;
    void setupDefaultGeometry();
    void setupDefaultOperatingConditions();
    void setupDefaultNumericalParameters();
};

} // namespace subchannel
} // namespace ants

#endif // ANTS_SUBCHANNEL_SOLVER_HPP
