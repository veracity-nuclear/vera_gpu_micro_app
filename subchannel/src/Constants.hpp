#ifndef ANTS_SUBCHANNEL_CONSTANTS_HPP
#define ANTS_SUBCHANNEL_CONSTANTS_HPP

namespace ants {
namespace subchannel {

// Mathematical constants
constexpr double PI = 3.14159265358979323846;

// Physical constants
constexpr double GRAVITY_ACCEL = 9.80665;           // gravity acceleration [m/s^2]
constexpr double CRITICAL_PRESSURE = 22.09e6;       // critical pressure [Pa]
constexpr double CONDENSATION_COEFF = 0.075;        // condensation coefficient [(1/degK)*(1/s)]

// ANTS Theory constants (from mechanistic models)
constexpr double ANTS_B1 = 1.5;                     // Chexal-Lellouche drift-flux constant
constexpr double ANTS_B2 = 1.41;                    // Chexal-Lellouche drift-flux constant
constexpr double ANTS_H0 = 0.075;                   // Condensation parameter [s^-1 K^-1]
constexpr double ANTS_THETA_M = 5.0;                // Two-phase eddy multiplier
constexpr double ANTS_K_M = 1.4;                    // Void drift scaling parameter
constexpr double ANTS_FRICTION_A1 = 0.1892;         // Blasius friction factor coefficient
constexpr double ANTS_FRICTION_N = 0.2;             // Blasius friction factor exponent

// Conversion factors
constexpr double PBR2SI = 6894.76;                  // [lbf/in^2] -> [Pa]
constexpr double VBR2SI = 0.062429767;              // [ft^3/lbm] -> [m^3/kg]
constexpr double EBR2SI = 2326.0;                   // [BTU/lbm]  -> [J/kg]
constexpr double SBR2SI = 14.59383202;              // [lbf/ft]   -> [N/m]
constexpr double QBR2SI = 0.29307;                  // [BTU/h]    -> [W]
constexpr double GBR2SI = 125.998;                  // [Mlb/h]    -> [kg/s]
constexpr double ABR2SI = 0.092903;                 // [ft^2]     -> [m^2]
constexpr double DBR2SI = 0.3048;                   // [ft]       -> [m]
constexpr double FBR2SI = 3.154;                    // [BTU/hr/ft^2] -> [W/m^2]
constexpr double MBR2SI = 0.45359291;               // [lbm] -> [kg]

// Numerical tolerances
constexpr double EPS = 1.0e-7;                      // General small number
constexpr double ETOL = 1.0e-8;                     // Iteration tolerance
constexpr double ZERO = 1.0e-14;                    // Essentially zero

// Water property parameters
constexpr double PC = 22.089e6;                     // Critical pressure [Pa]
constexpr double TC = 647.286;                      // Critical temperature [K]
constexpr double RHOC = 317.0;                      // Critical density [kg/m^3]

// Maximum array sizes
constexpr int TABMAX = 300;
constexpr int NMAXITS = 20;

} // namespace subchannel
} // namespace ants

#endif // ANTS_SUBCHANNEL_CONSTANTS_HPP
