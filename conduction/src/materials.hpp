#pragma once

#include <stdexcept>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>
#include "cylinder_node.hpp"

/*
Correlations for temperature-dependent properties are taken from the CTF v4.4 Manual
https://info.ornl.gov/sites/publications/Files/Pub203334.pdf

Materials:
  Solid:
    - Zircaloy
    - Fuel:
      - UO2
  Fluid:
    - Argon
    - Helium
    - Hydrogen
    - Krypton
    - Nitrogen
    - Xenon
*/

/**
 * @brief Base class for all materials.
 */
class Material {
public:
    /**
     * @brief Default constructor for Material.
     * @param name Name of the material.
     */
    explicit Material(const std::string& name) : name(name) {}

    /**
     * @brief Virtual destructor for Material.
     */
    virtual ~Material() = default;

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
 * @brief Abstract base class representing a solid material with temperature-dependent properties.
 */
class Solid : public Material {
public:
    /**
     * @brief Default constructor for Solid.
     * @param name Name of the material.
    */
    explicit Solid(const std::string& name) : Material(name) {}

    /**
     * @brief Update the radii of a CylinderNode based on temperature.
     * @param node CylinderNode to update.
     * @param T Current temperature in Kelvin.
     * @param T_prev Temperature from the previous calculation in Kelvin.
     * @note This method currently only accounts for thermal expansion. Relocation due to cracking,
     *  fuel burnup-induced swelling, clad burnup-induced creep, and clad elastic expansion are not considered.
     */
    virtual void update_node_radii(const std::shared_ptr<CylinderNode>& node, double T, double T_prev) const = 0;
};


/**
 * @brief Zircaloy (Zr2/4) material properties from CTF manual.
 */
class Zircaloy : public Solid {
public:
    /**
     * @brief Default constructor for Zircaloy.
     */
    Zircaloy() : Solid("Zr2/4") {}

    /**
     * @brief Constructor for Zircaloy with a custom name.
     * @param name Name of the material.
     */
    Zircaloy(const std::string &name) : Solid(name) {}

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

    /**
     * @brief Update the radii of a CylinderNode based on temperature.
     * @param node CylinderNode to update.
     * @param T Current temperature in Kelvin.
     * @param T_prev Previous temperature in Kelvin.
     * @note This method currently only accounts for thermal expansion. Relocation due to cracking,
     *  fuel burnup-induced swelling, clad burnup-induced creep, and clad elastic expansion are not considered.
     */
    void update_node_radii(const std::shared_ptr<CylinderNode>& node, double T, double T_prev) const override;
};


/**
 * @brief Abstract base class for fuel materials with additional burnup and gadolinium dependence.
 */
class Fuel : public Solid {
public:
    /**
     * @brief Default constructor for Fuel.
     */
    explicit Fuel() : Solid("Fuel") {}

    /**
     * @brief Constructor for Fuel with a custom name.
     * @param name Name of the material.
     */
    explicit Fuel(const std::string &name) : Solid(name) {}

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
 * @brief Uranium Dioxide (UO2) fuel material properties from CTF manual.
 */
class UO2 : public Fuel {
public:
    /**
     * @brief Default constructor for UO2.
     */
    explicit UO2() : Fuel("UO2") {}

    /**
     * @brief Constructor for UO2 with a custom name.
     * @param name Name of the material.
     */
    explicit UO2(const std::string &name) : Fuel(name) {}

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

    /**
     * @brief Update the radii of a CylinderNode based on temperature.
     * @param node CylinderNode to update.
     * @param T Current temperature in Kelvin.
     * @param T_prev Previous temperature in Kelvin.
     * @note This method currently only accounts for thermal expansion. Relocation due to cracking,
     *  fuel burnup-induced swelling, clad burnup-induced creep, and clad elastic expansion are not considered.
     */
    void update_node_radii(const std::shared_ptr<CylinderNode>& node, double T, double T_prev) const override;
};


/**
 * @brief Abstract base class for fluid materials with temperature-dependent properties.
 */
class Fluid : public Material {
public:
    /**
     * @brief Default constructor for Fluid.
     * @param name Name of the material.
     */
    explicit Fluid(const std::string &name) : Material(name) {}
};


/**
 * @brief Argon material properties from the CTF manual.
 */
class Argon : public Fluid {
public:
    /**
     * @brief Default constructor for Argon.
     */
    explicit Argon() : Fluid("Argon") {}

    /**
     * @brief Constructor for Argon with a custom name.
     * @param name Name of the material.
     */
    explicit Argon(const std::string &name) : Fluid(name) {}

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
    double Cp(double T) const override { throw std::runtime_error("Argon heat capacity not implemented"); }

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { throw std::runtime_error("Argon density not implemented"); }
};

/**
 * @brief Helium material properties from the CTF manual.
 */
class Helium : public Fluid {
public:
    /**
     * @brief Default constructor for Helium.
     */
    explicit Helium() : Fluid("Helium") {}

    /**
     * @brief Constructor for Helium with a custom name.
     * @param name Name of the material.
     */
    explicit Helium(const std::string &name) : Fluid(name) {}

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
 * @brief Hydrogen material properties from the CTF manual.
 */
class Hydrogen : public Fluid {
public:
    /**
     * @brief Default constructor for Hydrogen.
     */
    explicit Hydrogen() : Fluid("Hydrogen") {}

    /**
     * @brief Constructor for Hydrogen with a custom name.
     * @param name Name of the material.
     */
    explicit Hydrogen(const std::string &name) : Fluid(name) {}

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
    double Cp(double T) const override { throw std::runtime_error("Hydrogen heat capacity not implemented"); }

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { throw std::runtime_error("Hydrogen density not implemented"); }
};


/**
 * @brief Krypton material properties from the CTF manual.
 */
class Krypton : public Fluid {
public:
    /**
     * @brief Default constructor for Krypton.
     */
    explicit Krypton() : Fluid("Krypton") {}

    /**
     * @brief Constructor for Krypton with a custom name.
     * @param name Name of the material.
     */
    explicit Krypton(const std::string &name) : Fluid(name) {}

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
    double Cp(double T) const override { throw std::runtime_error("Krypton heat capacity not implemented"); }

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { throw std::runtime_error("Krypton density not implemented"); }
};


/**
 * @brief Nitrogen material properties from the CTF manual.
 */
class Nitrogen : public Fluid {
public:
    /**
     * @brief Default constructor for Nitrogen.
     */
    explicit Nitrogen() : Fluid("Nitrogen") {}

    /**
     * @brief Constructor for Nitrogen with a custom name.
     * @param name Name of the material.
     */
    explicit Nitrogen(const std::string &name) : Fluid(name) {}

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
    double Cp(double T) const override { throw std::runtime_error("Nitrogen heat capacity not implemented"); }

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { throw std::runtime_error("Nitrogen density not implemented"); }
};


/**
 * @brief Xenon material properties from the CTF manual.
 */
class Xenon : public Fluid {
public:
    /**
     * @brief Default constructor for Xenon.
     */
    explicit Xenon() : Fluid("Xenon") {}

    /**
     * @brief Constructor for Xenon with a custom name.
     * @param name Name of the material.
     */
    explicit Xenon(const std::string &name) : Fluid(name) {}

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
    double Cp(double T) const override { throw std::runtime_error("Xenon heat capacity not implemented"); }

    /**
     * @brief Get the temperature-dependent density. TODO: Implement this method.
     * @param T Temperature in Kelvin.
     * @return Density in kg/m³.
     */
    double rho(double T) const override { throw std::runtime_error("Xenon density not implemented"); }
};
