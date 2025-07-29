#pragma once

#include <iostream>
#include <vector>
#include <cmath>


/**
 * @brief Class representing a cylindrical node for heat conduction analysis.
 */
class CylinderNode {
public:

    /**
     * @brief Constructor for CylinderNode.
     * @param height Height of the cylinder in meters.
     * @param inner_radius Inner radius in meters.
     * @param outer_radius Outer radius in meters.
     */
    CylinderNode(double height, double inner_radius, double outer_radius);

    /**
     * @brief Get the inner radius of the cylinder node.
     * @return Inner radius in meters.
     */
    double get_inner_radius() const { return r_in; }

    /**
     * @brief Set the inner radius of the cylinder node.
     * @param inner_radius New inner radius in meters.
     */
    void set_inner_radius(double inner_radius);

    /**
     * @brief Get the outer radius of the cylinder node.
     * @return Outer radius in meters.
     */
    double get_outer_radius() const { return r_out; }

    /**
     * @brief Set the outer radius of the cylinder node.
     * @param outer_radius New outer radius in meters.
     */
    void set_outer_radius(double outer_radius);

    /**
     * @brief Get the height of the cylinder node.
     * @return Height in meters.
     */
    double get_height() const { return h; }

    /**
     * @brief Set the height of the cylinder node.
     * @param height New height in meters.
     */
    void set_height(double height);

    /**
     * @brief Get the initial inner radius of the cylinder node.
     * @return Initial inner radius in meters.
     */
    double get_initial_inner_radius() const { return r_in_initial; }

    /**
     * @brief Get the initial outer radius of the cylinder node.
     * @return Initial outer radius in meters.
     */
    double get_initial_outer_radius() const { return r_out_initial; }

    /**
     * @brief Get the inner surface area of the cylinder node.
     * @return Inner surface area in square meters.
     */
    double get_inner_area() const { return 2 * M_PI * r_in * h; }

    /**
     * @brief Get the outer surface area of the cylinder node.
     * @return Outer surface area in square meters.
     */
    double get_outer_area() const { return 2 * M_PI * r_out * h; }

    /**
     * @brief Get the volume of the cylinder node.
     * @return Volume in cubic meters.
     */
    double get_volume() const { return M_PI * h * (r_out * r_out - r_in * r_in); }

    /**
     * @brief Get the temperature of the cylinder node.
     * @return Temperature in Kelvin.
     */
    double get_temperature() const { return temperature; }

    /**
     * @brief Set the temperature of the cylinder node.
     * @param T Temperature in Kelvin.
     */
    void set_temperature(double T);

    /**
     * @brief Calculate the average temperature at the center of the cylinder node.
     * @param k Thermal conductivity in W/m-K.
     * @param T_inner Temperature at the inner surface in Kelvin.
     * @param T_outer Temperature at the outer surface in Kelvin.
     * @param qdot Heat generation rate in W/m³.
     * @return Average temperature at the center in Kelvin.
     */
    double calculate_avg_temperature(double k, double T_inner, double T_outer, double qdot) const;

    /**
     * @brief Solve for the inner temperature of the cylinder node.
     * @param k Thermal conductivity in W/m-K.
     * @param T_outer Temperature at the outer surface in Kelvin.
     * @param qflux Heat flux in W/m².
     * @param qdot Heat generation rate in W/m³.
     * @return Inner temperature in Kelvin.
     */
    double solve_inner_temperature(double k, double T_outer, double qflux, double qdot) const;

    /**
     * @brief Calculate the temperature at the inner surface of a cylinder with no heat generation.
     * @param k Thermal conductivity in W/m-K.
     * @param r_out Outer radius in meters.
     * @param T_outer Temperature at the outer surface in Kelvin.
     * @param qdot Heat generation rate in W/m³.
     * @return Temperature at the inner surface in Kelvin.
     */
    double cyl_Tin(double k, double r_out, double T_outer, double qdot) const;

    /**
     * @brief Calculate the temperature at the inner surface of a cylindrical shell with no heat generation.
     * @param k Thermal conductivity in W/m-K.
     * @param r_in Inner radius in meters.
     * @param r_out Outer radius in meters.
     * @param T_outer Temperature at the outer surface in Kelvin.
     * @param qflux Heat flux in W/m².
     * @return Temperature at the inner surface in Kelvin.
     */
    double cyl_shell_Tin(double k, double r_in, double r_out, double T_outer, double qflux) const;

    /**
     * @brief Calculate the temperature at the inner surface of a cylindrical shell with heat generation.
     * @param k Thermal conductivity in W/m-K.
     * @param r_in Inner radius in meters.
     * @param r_out Outer radius in meters.
     * @param T_outer Temperature at the outer surface in Kelvin.
     * @param qflux Heat flux in W/m².
     * @param qdot Heat generation rate in W/m³.
     * @return Temperature at the inner surface in Kelvin.
     */
    double cyl_shell_Tin_heat_gen(double k, double r_in, double r_out, double T_outer, double qflux, double qdot) const;

private:
    /**
     * @brief Height of the cylinder node.
     */
    double h;

    /**
     * @brief Inner radius of the cylinder node.
     */
    double r_in;

    /**
     * @brief Outer radius of the cylinder node.
     */
    double r_out;

    /**
     * @brief Initial inner radius of the cylinder node.
     */
    double r_in_initial;

    /**
     * @brief Initial outer radius of the cylinder node.
     */
    double r_out_initial;

    /**
     * @brief Temperature of the cylinder node in Kelvin.
     */
    double temperature;
};
