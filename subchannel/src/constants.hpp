#pragma once

// Mathematical constants
const double PI = 3.14159265358979323846;

// Physical constants
const double GRAVITY_ACCEL = 9.80665;           // gravity acceleration [m/s^2]
const double CRITICAL_PRESSURE = 22.09e6;       // critical pressure [Pa]
const double CONDENSATION_COEFF = 0.075;        // condensation coefficient [(1/degK)*(1/s)]

// ANTS Theory constants (from mechanistic models)
const double ANTS_B1 = 1.5;                     // Chexal-Lellouche drift-flux constant
const double ANTS_B2 = 1.41;                    // Chexal-Lellouche drift-flux constant
const double ANTS_H0 = 0.075;                   // Condensation parameter [s^-1 K^-1]
const double ANTS_THETA_M = 5.0;                // Two-phase eddy multiplier
const double ANTS_K_M = 1.4;                    // Void drift scaling parameter
const double ANTS_FRICTION_A1 = 0.1892;         // Blasius friction factor coefficient
const double ANTS_FRICTION_N = 0.2;             // Blasius friction factor exponent

// Conversion factors
const double PBR2SI = 6894.76;                  // [lbf/in^2] -> [Pa]
const double VBR2SI = 0.062429767;              // [ft^3/lbm] -> [m^3/kg]
const double EBR2SI = 2326.0;                   // [BTU/lbm]  -> [J/kg]
const double SBR2SI = 14.59383202;              // [lbf/ft]   -> [N/m]
const double QBR2SI = 0.29307;                  // [BTU/h]    -> [W]
const double GBR2SI = 125.998;                  // [Mlb/h]    -> [kg/s]
const double ABR2SI = 0.092903;                 // [ft^2]     -> [m^2]
const double DBR2SI = 0.3048;                   // [ft]       -> [m]
const double FBR2SI = 3.154;                    // [BTU/hr/ft^2] -> [W/m^2]
const double MBR2SI = 0.45359291;               // [lbm] -> [kg]

// Numerical tolerances
const double EPS = 1.0e-7;                      // General small number
const double ETOL = 1.0e-8;                     // Iteration tolerance
const double ZERO = 1.0e-14;                    // Essentially zero

// Water property parameters
const double PC = 22.089e6;                     // Critical pressure [Pa]
const double TC = 647.286;                      // Critical temperature [K]
const double RHOC = 317.0;                      // Critical density [kg/m^3]
