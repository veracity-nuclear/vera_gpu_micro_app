#pragma once

#include <Kokkos_Core.hpp>

struct Water {

    // enthalpy [J/kg]
    KOKKOS_INLINE_FUNCTION constexpr double h(double T) const { return 4220.0 * (T - 273.15); }
    KOKKOS_INLINE_FUNCTION constexpr double h_f() const { return 1263.1e3; } // saturated liquid enthalpy at saturation temperature
    KOKKOS_INLINE_FUNCTION constexpr double h_g() const { return 2773.7e3; } // saturated vapor enthalpy at saturation temperature
    KOKKOS_INLINE_FUNCTION constexpr double h_fg() const { return h_g() - h_f(); } // latent heat of vaporization at saturation temperature

    // temperature [K]
    KOKKOS_INLINE_FUNCTION constexpr double T(double h) const { return h / 4220.0 + 273.15; }
    KOKKOS_INLINE_FUNCTION constexpr double Tsat() const { return 285.83 + 273.15; } // K, saturation temperature at 7 MPa

    // density [kg/m^3]
    KOKKOS_INLINE_FUNCTION constexpr double rho(double /*h*/) const { return 958.0; }
    KOKKOS_INLINE_FUNCTION constexpr double rho_f() const { return 1 / v_f(); } // saturated liquid density at saturation temperature
    KOKKOS_INLINE_FUNCTION constexpr double rho_g() const { return 1 / v_g(); } // saturated vapor density at saturation temperature
    KOKKOS_INLINE_FUNCTION constexpr double rho_m(double X) const { return 1 / (X * v_g() + (1 - X) * v_f()); } // mixture density at saturation temperature

    // specific volume [m^3/kg]
    KOKKOS_INLINE_FUNCTION constexpr double v_f() const { return 0.001349; } // saturated liquid specific volume at saturation temperature
    KOKKOS_INLINE_FUNCTION constexpr double v_g() const { return 0.027756; } // saturated vapor specific volume at saturation temperature
    KOKKOS_INLINE_FUNCTION constexpr double v_fg() const { return v_g() - v_f(); } // difference in specific volume at saturation temperature

    // specific heat [J/kg-K]
    KOKKOS_INLINE_FUNCTION constexpr double Cp(double /*h*/) const { return 4220.0; }

    // viscosity [Pa-s]
    KOKKOS_INLINE_FUNCTION constexpr double mu(double /*h*/) const { return 0.001352; }
    KOKKOS_INLINE_FUNCTION constexpr double mu_f() const { return 91.266e-6; } // saturated liquid viscosity at saturation temperature
    KOKKOS_INLINE_FUNCTION constexpr double mu_g() const { return 18.890e-6; } // saturated vapor viscosity at saturation temperature

    // thermal conductivity [W/m-K]
    KOKKOS_INLINE_FUNCTION constexpr double k(double /*h*/) const { return 0.6; }

    // surface tension [N/m]
    KOKKOS_INLINE_FUNCTION constexpr double sigma() const { return 0.02; } // N/m, approximate value for water
};
