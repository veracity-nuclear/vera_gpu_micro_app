#pragma once

#include <iostream>
#include <vector>
#include <cmath>

#define PI M_PI

class CylinderNode {
public:
    CylinderNode(double height, double inner_radius, double outer_radius)
        : h(height), r_in(inner_radius), r_out(outer_radius) {
            if (height <= 0) {
                throw std::invalid_argument("Height must be positive for CylinderNode");
            }
            if (r_in < 0 || r_out <= 0 || r_in >= r_out) {
                throw std::invalid_argument("Invalid radius values for CylinderNode");
            }
        }

    double get_inner_radius() const { return r_in; }
    void set_inner_radius(double inner_radius) {
        if (inner_radius < 0 || inner_radius >= r_out) {
            throw std::invalid_argument("Invalid inner radius value");
        }
        r_in = inner_radius;
    }

    double get_outer_radius() const { return r_out; }
    void set_outer_radius(double outer_radius) {
        if (outer_radius <= 0 || outer_radius <= r_in) {
            throw std::invalid_argument("Invalid outer radius value");
        }
        r_out = outer_radius;
    }

    double get_height() const { return h; }
    void set_height(double height) {
        if (height <= 0) {
            throw std::invalid_argument("Height must be positive");
        }
        h = height;
    }

    double get_inner_area() const { return 2 * PI * r_in * h; }
    double get_outer_area() const { return 2 * PI * r_out * h; }
    double get_volume() const { return PI * h * (r_out * r_out - r_in * r_in); }
    double get_temperature() const { return temperature; }
    void set_temperature(double temp) {
        if (temp < 0.0) {
            throw std::out_of_range("Temperature in Kelvin cannot be less than 0.0");
        }
        temperature = temp;
    }

    double calculate_thermal_resistance(double k) const {
        if (r_in == 0) {
            return 1 / (4 * PI * k * h);
        }
        return log(r_out / r_in) / (2 * PI * k * h);
    }

private:
    double h;
    double r_in;
    double r_out;
    double temperature = 600.0; // Temperature in Kelvin, to be set by the solver
};
