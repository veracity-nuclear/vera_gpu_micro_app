#include "cylinder_node.hpp"

CylinderNode::CylinderNode(double height, double inner_radius, double outer_radius)
    : h(height), r_in(inner_radius), r_out(outer_radius), r_in_initial(inner_radius), r_out_initial(outer_radius), temperature(600.0) {
        if (height <= 0) {
            throw std::invalid_argument("Height must be positive for CylinderNode");
        }
        if (r_in < 0 || r_out <= 0 || r_in >= r_out) {
            throw std::invalid_argument("Invalid radius values for CylinderNode");
        }
    }

void CylinderNode::set_inner_radius(double inner_radius) {
    if (inner_radius < 0 || inner_radius >= r_out) {
        throw std::invalid_argument("Invalid inner radius value");
    }
    r_in = inner_radius;
}

void CylinderNode::set_outer_radius(double outer_radius) {
    if (outer_radius <= 0 || outer_radius <= r_in) {
        throw std::invalid_argument("Invalid outer radius value");
    }
    r_out = outer_radius;
}

void CylinderNode::set_height(double height) {
    if (height <= 0) {
        throw std::invalid_argument("Height must be positive");
    }
    h = height;
}

void CylinderNode::set_temperature(double T) {
    if (T < 0.0) {
        throw std::out_of_range("Temperature in Kelvin cannot be less than 0.0");
    }
    temperature = T;
}

double CylinderNode::calculate_avg_temperature(double k, double T_inner, double T_outer, double qdot) const {

    // Convert from W/m-K to W/cm-K
    k *= 100.0; // W/cm-K

    double r_avg = (r_in + r_out) * 0.5;

    if (r_in < 1e-9) {
        return T_outer + 3.0 / 16.0 * (qdot / k) * r_out * r_out;
    } else {
        // Solve for C1
        double num = (T_outer - T_inner) + (qdot / (4.0 * k)) * (r_out * r_out - r_in * r_in);
        double denom = std::log(r_out) - std::log(r_in);
        double C1 = num / denom;

        // Solve for C2
        double C2 = T_inner + (qdot / (4.0 * k)) * r_in * r_in - C1 * std::log(r_in);

        // Evaluate T at average radius
        return -(qdot / (4.0 * k)) * r_avg * r_avg + C1 * std::log(r_avg) + C2;
    }
}

double CylinderNode::solve_inner_temperature(double k, double T_outer, double qflux, double qdot) const {

    // convert from W/m-K to W/cm-K
    k *= 100.0; // W/cm-K

    if (r_in < 1e-9) {
        return cyl_Tin(k, r_out, T_outer, qdot);
    }
    if (qdot == 0.0) {
        return cyl_shell_Tin(k, r_in, r_out, T_outer, qflux);
    }
    return cyl_shell_Tin_heat_gen(k, r_in, r_out, T_outer, qflux, qdot);
}

double CylinderNode::cyl_Tin(double k, double r_out, double T_outer, double qdot) const {
    return T_outer + qdot * r_out * r_out / (4 * k);
}

double CylinderNode::cyl_shell_Tin(double k, double r_in, double r_out, double T_outer, double qflux) const {
    double C1 = qflux * r_in / k;
    double C2 = T_outer + C1 * std::log(r_out);
    return C2 - C1 * std::log(r_in);
}

double CylinderNode::cyl_shell_Tin_heat_gen(double k, double r_in, double r_out, double T_outer, double qflux, double qdot) const {
    double C0 = -qdot / (4.0 * k);
    double C1 = (qflux * r_in - 0.5 * qdot * r_in * r_in) / k;
    double C2 = T_outer + C1 * std::log(r_out) - C0 * r_out * r_out;
    return C2 - C1 * std::log(r_in) + C0 * r_in * r_in;
}
