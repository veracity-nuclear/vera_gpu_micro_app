#ifndef ANTS_SUBCHANNEL_THERMAL_HYDRAULICS_HPP
#define ANTS_SUBCHANNEL_THERMAL_HYDRAULICS_HPP

#include "Constants.hpp"
#include "SubchannelData.hpp"
#include "Properties.hpp"
#include <memory>

namespace ants {
namespace subchannel {

/**
 * @brief Main thermal hydraulics solver for subchannel analysis
 *
 * This class implements the ANTS thermal hydraulics solver, including:
 * - Conservation equations for liquid mass, vapor mass, and energy
 * - Axial momentum equation with two-phase pressure drop
 * - Boiling and condensation models
 * - Drift-flux void fraction model
 */
class ThermalHydraulics {
public:
    ThermalHydraulics(std::shared_ptr<SubchannelData> data,
                      std::shared_ptr<Properties> properties);
    ~ThermalHydraulics() = default;

    // Main solving interface
    void solve();
    void solveSingleChannel(int channel_id);
    void solveMultiChannel();

    // Individual equation solvers
    void solveAxialMarch(int channel_id);
    void solveCrossflowIteration();

    // Physics models
    void calculateBoiling(int channel_id, int axial_node);
    void calculateEvaporation(int channel_id, int axial_node);
    void calculateVoidFraction(int channel_id, int axial_node);
    void calculatePressureDrop(int channel_id, int axial_node);

    // Property calculations
    void updateFlowProperties(int channel_id, int axial_node);
    void calculateEnthalpyAndQuality(int channel_id, int axial_node);

    // Convergence and iteration control
    bool checkConvergence() const;
    void relaxSolution(double relaxation_factor);

    // Getters
    double getExitVoidFraction(int channel_id = 0) const;
    double getExitTemperature(int channel_id = 0) const;  // Returns in Celsius
    double getExitTemperatureK(int channel_id = 0) const; // Returns in Kelvin
    double getPressureDrop(int channel_id = 0) const;

    // Results output
    void printSolution() const;
    void printChannelResults(int channel_id) const;

private:
    std::shared_ptr<SubchannelData> data_;
    std::shared_ptr<Properties> properties_;

    // Working arrays for iteration
    std::vector<double> old_crossflow_;
    std::vector<double> residual_;

    // Physical constants and correlations
    double viscosity_liquid_;
    double thermal_conductivity_;
    double prandtl_number_;
    double reynolds_number_;
    double peclet_number_;

    // Two-phase correlations
    double chisholm_parameter_b_;
    double gamma_squared_;

    // Helper functions
    int getAxialIndex(int channel_id, int axial_node) const;
    int getCellIndex(int channel_id, int axial_node) const;

    // Equation solvers
    void solveEnthalpyEquation(int channel_id, int axial_node,
                               double inlet_liquid_flow, double inlet_vapor_flow,
                               double inlet_mixture_enthalpy, double heat_input,
                               double& outlet_liquid_flow, double& outlet_vapor_flow,
                               double& outlet_mixture_enthalpy, double& outlet_liquid_enthalpy,
                               double& outlet_flow_quality, double& outlet_liquid_density);

    void solveVoidFractionEquation(int channel_id, int axial_node,
                                   double liquid_flow, double vapor_flow,
                                   double liquid_density, double& void_fraction);

    void calculateAxialPressureDrop(int channel_id, int axial_node,
                                    double liquid_flow, double vapor_flow,
                                    double void_fraction, double liquid_density,
                                    double flow_quality, double momentum_flux_out,
                                    double momentum_flux_in, double crossflow_momentum,
                                    double& pressure_drop);

    // Boiling model functions
    double calculateDepartureEnthalpy(double mass_flux, double heat_flux) const;
    double calculateCondensationRate(double void_fraction, double subcooling) const;
    double calculateEvaporationRate(double heat_flux, double subcooling,
                                   double departure_enthalpy) const;

    // Two-phase flow correlations
    double calculateHomogeneousMultiplier(double flow_quality,
                                         double liquid_density, double vapor_density) const;
    double calculateChisholmMultiplier(double flow_quality, double mass_flux) const;
    double calculateFrictionFactor(double reynolds_number) const;
    double calculateDistributionParameter(double void_fraction) const;
    double calculateDriftVelocity(double void_fraction) const;

    // Numerical methods
    void updatePropertiesAtNode(int channel_id, int axial_node);
    double calculateReynoldsNumber(double mass_flux, double hydraulic_diameter,
                                  double viscosity) const;

    // Iteration tracking
    int getNewtonIterationCount() const;
    void incrementNewtonIterationCount();
    void resetNewtonIterationCount();

    // ANTS Transverse momentum solver
    void solveCrossflowAtAxialPlane(int axial_node);
    void calculateTransverseMomentumResiduals(int axial_node,
                                            const std::vector<double>& G_CF,
                                            std::vector<double>& residual);
    void buildTransverseMomentumJacobian(int axial_node,
                                       const std::vector<double>& G_CF,
                                       std::vector<std::vector<double>>& jacobian);
    double calculateMixtureVelocity(int surface_id, int axial_node, double G_CF);
    void getSurfaceChannels(int surface_id, int& ch_i, int& ch_j);
    void solveLinearSystem(const std::vector<std::vector<double>>& A,
                         const std::vector<double>& b,
                         std::vector<double>& x);
    int getSurfaceIndex(int surface_id, int axial_node) const;
};

} // namespace subchannel
} // namespace ants

#endif // ANTS_SUBCHANNEL_THERMAL_HYDRAULICS_HPP
