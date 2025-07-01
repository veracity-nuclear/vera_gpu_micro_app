#pragma once

#include <stdexcept>

/**
 * @brief Abstract base class representing a solid material with temperature-dependent properties.
 */
class SolidMaterial {
public:
    virtual ~SolidMaterial() = default;

    /**
     * @brief Get the temperature-dependent thermal conductivity.
     * @param T Temperature in Kelvin.
     * @return Thermal conductivity in W/m-K.
     */
    virtual double k(double T) const = 0;

    /**
     * @brief Get the temperature-dependent specific heat capacity.
     * @param T Temperature in Kelvin.
     * @return Specific heat capacity in J/kg-K.
     */
    virtual double Cp(double T) const = 0;

    /**
     * @brief Get the temperature-dependent density.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    virtual double rho(double T) const = 0;
};

class Fuel : public SolidMaterial {
    public:
        Fuel() = default;
        double k(double T) const override { return 1.05 + 2150 / (T - 73.15); }
        double Cp(double T) const override { throw std::runtime_error("Fuel heat capacity not implemented"); }
        double rho(double T) const override { throw std::runtime_error("Fuel density not implemented"); }
};

class Clad : public SolidMaterial {
    public:
        Clad() = default;
        double k(double T) const override { return 7.51 + 2.09e-2 * T - 1.45e-5 * T * T + 7.67e-9 * T * T * T; }
        double Cp(double T) const override { throw std::runtime_error("Clad heat capacity not implemented"); }
        double rho(double T) const override { throw std::runtime_error("Clad density not implemented"); }
};
