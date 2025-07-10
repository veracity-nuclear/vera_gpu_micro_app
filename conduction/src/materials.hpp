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


/**
 * @brief Abstract base class for fuel materials with additional burnup and gadolinium dependence.
 */
class FuelMaterial : public SolidMaterial {
public:
    /**
     * @brief Default constructor for FuelMaterial.
     */
    explicit FuelMaterial() : SolidMaterial("Fuel") {}

    /**
     * @brief Constructor for FuelMaterial with a custom name.
     * @param name Name of the material.
     */
    explicit FuelMaterial(const std::string &name) : SolidMaterial(name) {}

    /**
     * @brief Get the temperature-dependent thermal conductivity.
     * @param T Temperature in Kelvin.
     * @return Thermal conductivity in W/m-K.
     */
    double k(double T) const override { return k(T, 0.0, 0.0); }

    /**
     * @brief Get the temperature-dependent specific heat capacity.
     * @param T Temperature in Kelvin.
     * @return Specific heat capacity in J/kg-K.
     */
    double Cp(double T) const override { return Cp(T, 0.0, 0.0); }

    /**
     * @brief Get the temperature-dependent density.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { return rho(T, 0.0, 0.0); }

    /**
     * @brief Get the thermal conductivity as a function of temperature, burnup and gadolinium content.
     * @param T Temperature in Kelvin.
     * @return Thermal conductivity in W/m-K.
     */
    virtual double k(double T, double Bu, double gad) const = 0;

    /**
     * @brief Get the specific heat capacity as a function of temperature, burnup and gadolinium content.
     * @param T Temperature in Kelvin.
     * @return Specific heat capacity in J/kg-K.
     */
    virtual double Cp(double T, double Bu, double gad) const = 0;

    /**
     * @brief Get the density as a function of temperature, burnup and gadolinium content.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    virtual double rho(double T, double Bu, double gad) const = 0;
};


/**
 * @brief Helium material properties from the CTF manual.
 */
class Helium : public SolidMaterial {
public:
    /**
     * @brief Default constructor for Helium.
     */
    explicit Helium() : SolidMaterial("Helium") {}

    /**
     * @brief Constructor for Helium with a custom name.
     * @param name Name of the material.
     */
    explicit Helium(const std::string &name) : SolidMaterial(name) {}

    /**
     * @brief Get the temperature-dependent thermal conductivity.
     * @param T Temperature in Kelvin.
     * @return Thermal conductivity in W/m-K.
     */
    double k(double T) const override;

    /**
     * @brief Get the temperature-dependent specific heat capacity. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Specific heat capacity in J/kg-K.
     */
    double Cp(double T) const override { throw std::runtime_error("Helium heat capacity not implemented"); }

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { throw std::runtime_error("Helium density not implemented"); }
};


/**
 * @brief Uranium Dioxide (UO2) fuel material properties from CTF manual.
 */
class UO2 : public FuelMaterial {
public:
    /**
     * @brief Default constructor for UO2.
     */
    explicit UO2() : FuelMaterial("UO2") {}

    /**
     * @brief Constructor for UO2 with a custom name.
     * @param name Name of the material.
     */
    explicit UO2(const std::string &name) : FuelMaterial(name) {}

    /**
     * @brief Get the temperature-dependent thermal conductivity.
     * @param T Temperature in Kelvin.
     * @param Bu Burnup in MWd/kgU.
     * @param gad Gadolinium content in wt. percent.
     * @return Thermal conductivity in W/m-K.
     */
    double k(double T, double Bu, double gad) const override;

    /**
     * @brief Get the temperature-dependent specific heat capacity.
     * @param T Temperature in Kelvin.
     * @param Bu Burnup in MWd/kgU.
     * @param gad Gadolinium content in wt. percent.
     * @return Specific heat capacity in J/kg-K.
     */
    double Cp(double T, double Bu, double gad) const override;

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @param Bu Burnup in MWd/kgU.
     * @param gad Gadolinium content in wt. percent.
     * @return Density in kg/m³.
     */
    double rho(double T, double Bu, double gad) const override { throw std::runtime_error("UO2 density not implemented"); }
};


/**
 * @brief Zircaloy (Zr2/4) material properties from CTF manual.
 */
class Zircaloy : public SolidMaterial {
public:
    /**
     * @brief Default constructor for Zircaloy.
     */
    Zircaloy() : SolidMaterial("Zr2/4") {}

    /**
     * @brief Constructor for Zircaloy with a custom name.
     * @param name Name of the material.
     */
    Zircaloy(const std::string &name) : SolidMaterial(name) {}

    /**
     * @brief Get the temperature-dependent thermal conductivity.
     * @param T Temperature in Kelvin.
     * @return Thermal conductivity in W/m-K.
     */
    double k(double T) const override;

    /**
     * @brief Get the temperature-dependent specific heat capacity.
     * @param T Temperature in Kelvin.
     * @return Specific heat capacity in J/kg-K.
     */
    double Cp(double T) const override;

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { throw std::runtime_error("Zircaloy density not implemented"); }
};

