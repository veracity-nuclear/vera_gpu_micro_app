# ANTS: Alternative Nonlinear Two-phase Subchannel Solver

## Overview

ANTS is a C++ implementation of a two-phase subchannel thermal hydraulic solver, converted from the original Fortran modules in the APEX code. The solver implements steady-state conservation equations for liquid mass, vapor mass, mixture energy, and axial mixture momentum with drift-flux void fraction correlations.

## Physics and Numerics

### Governing Equations
- Steady conservation of liquid mass, vapor mass, mixture energy, axial mixture momentum, and transverse mixture momentum (driven by pressure differences)
- Transverse system solved via non-linear outer iteration over a plane; axial variables solved via inner iteration with donor-cell upwinding for crossflow quantities
- Uses drift-flux closure (Chexal–Lellouche) for void fraction; two-phase pressure drop uses Chisholm-style multipliers for wall shear and geometric losses

### Iteration Scheme
- **Outer iteration**: Solve transverse momentum residuals for surface mixture mass flux using a Newton/line-search update on G_CF with numerically built Jacobian
- **Inner iteration**: Given surface sources, solve sequentially for W_l, W_v, h_l, alpha, then pressure drop in each axial march step from inlet to outlet
- **Donor cell**: Upwind (donor) enthalpy/velocity/quality across surfaces for crossflow terms

## Build Instructions

### Prerequisites
- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2017+)
- CMake 3.16 or higher

### Building

From the vera_gpu_micro_app root directory:

```bash
mkdir -p build
cd build
cmake ..
make ants_subchannel
make ants_3x3_verification
```

Or build everything:
```bash
make -j$(nproc)
```

### Building Tests

```bash
make subchannel_tests
make test  # Run all tests including ANTS verification
```

## Running the 3x3 Verification Case

### Execute the verification case:

```bash
cd build
./subchannel/ants_3x3_verification
```

### Expected Output

The solver will print detailed solution information and conclude with:

```
FINAL RESULTS (as requested):
============================================================
Total/average bundle pressure drop [Pa]: [value]
Exit void fraction (bundle-average) [-]: [value]  
Exit temperature [°C]: [value]
Exit temperature [K]: [value]
============================================================
```

Results are also written to `ants_3x3_results.txt`.

### Input Parameters (3x3 Verification Case)

The verification case uses the following parameters:

| Parameter | Value | Units |
|-----------|-------|-------|
| Flow area | 1.436 | cm² |
| Hydraulic diameter | 1.486 | cm |
| Height | 381.0 | cm |
| Gap width | 0.39 | cm |
| Linear heat rate | 29.1 | kW/m |
| Inlet temperature | 278.0 | °C |
| Inlet pressure | 7.255 | MPa |
| Inlet flow rate | 2.25 | kg/s |
| Gap loss Kns | 0.5 | - |

### Convergence Tolerances

- Outer residual (transverse momentum): 1e-6 (absolute)
- Inner residual (void): 1e-14 (absolute)

## Code Structure

### Core Classes

1. **Constants.hpp**: Physical constants and conversion factors
2. **Properties.hpp/.cpp**: Water/steam properties and drift-flux correlations  
3. **SubchannelData.hpp/.cpp**: Data structures for subchannel variables
4. **ThermalHydraulics.hpp/.cpp**: Main T/H solver with physics models
5. **Solver.hpp/.cpp**: Main solver interface and coordination

### Key Features

- **Modular design**: Clear separation of interfaces (hpp) and implementations (cpp)
- **SI units internally**: Converts for human-readable prints as required
- **Error handling**: Throws `std::runtime_error` with clear messages  
- **Minimal dependencies**: Uses only standard C++ library
- **One class per file**: Following the original Fortran module structure

## Verification and Testing

### 3x3 Pin-Centered Subchannel Case

The included verification case represents a 3x3 pin arrangement with pin-centered subchannel analysis. The unheated center subchannel evolves solely through pressure-directed crossflow, turbulent mixing, and void drift. The solution should converge under axial mesh refinement.

### Expected Behavior

- Pressure drop dominated by two-phase friction and form losses
- Void fraction development along heated length  
- Temperature rise consistent with energy balance
- Convergent solution under mesh refinement

## Limitations and Future Work

### Current Implementation

- Single representative subchannel (3x3 topology not fully implemented)
- Simplified property correlations (full IAPWS would be preferred)  
- No actual crossflow momentum solver (placeholder implementation)

### Future Enhancements

- Full 3x3 multichannel with crossflow momentum equation
- 10x10 BWR verification case
- Advanced turbulent mixing models
- Improved property correlations
- Parallel execution capabilities

## Conversion Notes

This C++ implementation was converted from the following Fortran modules in apex/src:

- `th_mod.f90` → `ThermalHydraulics.hpp/.cpp`
- `props_modb.f90` → `Properties.hpp/.cpp` 
- `util_mod.f90` → Integrated into `Properties.cpp`
- `flux_modb.f90` (subchannel portions) → `SubchannelData.hpp/.cpp`
- `solver_mod.f90` (subchannel portions) → `Solver.hpp/.cpp`
- `main.f90` (subchannel portions) → `verification_3x3.cpp`

Non-subchannel components (neutronics, burnup, cross-sections) were excluded from the conversion.

## License

[Include appropriate license information here]
