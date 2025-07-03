#include "materials.hpp"

double Clad::k(double T) const {
    if (T < 0.0) {
        throw std::out_of_range("Temperature in Kelvin cannot be less than 0.0");
    }

    return 7.51 + 2.09e-2 * T - 1.45e-5 * T * T + 7.67e-9 * T * T * T;
}

double Clad::Cp(double T) const override {
    static const std::vector<double> T_vals = {
        300.0, 400.0, 640.0, 1090.0, 1093.0, 1113.0, 1133.0, 1153.0, 1173.0,
        1193.0, 1213.0, 1233.0, 1248.0, 2098.0, 2099.0
    };
    static const std::vector<double> Cp_vals = {
        281.0, 302.0, 331.0, 375.0, 502.0, 590.0, 615.0, 719.0, 816.0,
        770.0, 619.0, 469.0, 356.0, 356.0, 356.0
    };

    if (T < T_vals.front() || T > T_vals.back()) {
        throw std::out_of_range("Temperature must be within 300.0 and 2100.0 K for Clad heat capacity");
    }

    auto upper = std::upper_bound(T_vals.begin(), T_vals.end(), T);
    size_t i = std::distance(T_vals.begin(), upper) - 1;

    double T0 = T[i], T1 = T[i + 1];
    double Cp0 = Cp_vals[i], Cp1 = Cp_vals[i + 1];

    return Cp0 + (Cp1 - Cp0) * (T - T0) / (T1 - T0);
}

double UO2::k(double T, double Bu, double gad) const {
    if (T < 300.0 || T > 3000.0) {
        throw std::out_of_range("Temperature must be within 300.0 and 3000.0 K for UO2 thermal conductivity");
    }
    if (Bu < 0.0 || Bu > 62.0) {
        throw std::out_of_range("Burnup must be between 0.0 and 62.0 MWd/kgU for UO2 thermal conductivity");
    }
    if (gad < 0.0 || gad > 0.1) {
        throw std::out_of_range("Gadolinium content must be between 0.0 and 0.10 wt. percent for UO2 thermal conductivity");
    }

    double h = 1 / (1.0 + 396.0 * std::exp(-6380.0 / T));
    double k_phonon = 1 / (0.0452 + 0.000246 * T + 0.00187 * Bu + 1.1599 * gad + (1.0 - 0.9 * std::exp(-0.04 * Bu)) \
        * 0.038 * h * std::pow(Bu, 0.28));
    double k_electronic = 3.50e9 / std::pow(T, 2.0) * std::exp(-16361.0 / T);

    return k_phonon + k_electronic;
}

double UO2::Cp(double T, double Bu, double gad) const {
    double OM = 2.0; // oxygen-to-metal ratio for UO2
    double R = 8.314; // universal gas constant [J/mol-K]
    double theta = 535.285; // Einstein temperature [K]
    double E_D = 1.577e5; // Debye energy [J/mol]
    double K1, K2, K3 = 296.7, 2.43e-2, 8.745e7; // empirical constants for UO2 heat capacity [J/kg-K]

    return K1 * std::pow(theta, 2.0) * std::exp(theta / T) / (std::pow(T, 2.0) * (std::exp(theta / T) - 1.0)) \
        + K2 * T + OM / 2 * K3 * E_D / (R * std::pow(T, 2.0)) * std::exp(-E_D / (R * T));
}
