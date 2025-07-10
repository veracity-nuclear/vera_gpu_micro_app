#pragma once

#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

/**
 * @brief Abstract base class representing a solid material with temperature-dependent properties.
 */
class SolidMaterial {
public:
    /**
     * @brief Default constructor for SolidMaterial.
     * @param name Name of the material.
    */
    explicit SolidMaterial(const std::string& name) : name(name) {}

    /**
     * @brief Virtual destructor for SolidMaterial.
     */
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

    /**
     * @brief Get the name of the material.
     * @return Name of the material.
     */
    std::string getName() const { return name; }

protected:
    /**
     * @brief Name of the material, used for identification and debugging.
     */
    std::string name;
};

class Clad : public SolidMaterial {
public:
    Clad() : SolidMaterial("Clad") {}
    Clad(const std::string &name) : SolidMaterial(name) {}
    double k(double T) const override;
    double Cp(double T) const override;
    double rho(double T) const override { throw std::runtime_error("Clad density not implemented"); }
};

class Gap : public SolidMaterial {
public:
    Gap() : SolidMaterial("Gap") {}
    Gap(const std::string &name) : SolidMaterial(name) {}
    double k(double T) const override { return 0.2; }
    double Cp(double T) const override { throw std::runtime_error("Gap heat capacity not implemented"); }
    double rho(double T) const override { throw std::runtime_error("Gap density not implemented"); }
};

class FuelMaterial : public SolidMaterial {
public:
    FuelMaterial() : SolidMaterial("Fuel") {}
    FuelMaterial(const std::string &name) : SolidMaterial(name) {}
    virtual double k(double T, double Bu, double gad) const = 0;
    virtual double Cp(double T, double Bu, double gad) const = 0;
    virtual double rho(double T, double Bu, double gad) const = 0;
};

class UO2 : public FuelMaterial {
public:
    UO2() : FuelMaterial("UO2") {}
    UO2(const std::string &name) : FuelMaterial(name) {}

    // pure virtuals from SolidMaterial with default burnup/gadolinium
    double k(double T) const override { return k(T, 0.0, 0.0); }
    double Cp(double T) const override { return Cp(T, 0.0, 0.0); }
    double rho(double T) const override { return rho(T, 0.0, 0.0); }

    // methods w/ burnup and gadolinium dependence for full solver use
    double k(double T, double Bu, double gad) const override;
    double Cp(double T, double Bu, double gad) const override;
    double rho(double T, double Bu, double gad) const override { throw std::runtime_error("UO2 density not implemented"); }
};

