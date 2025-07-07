#pragma once
#include <string>
#include <vector>

constexpr int RAY_START = 0; // Index for the start of the ray
constexpr int RAY_END = 1;   // Index for the end of the ray

class BaseMOC {
public:
    // Constructor
    BaseMOC() = default;

    // Virtual destructor
    virtual ~BaseMOC() = default;

    // Run the MOC sweep
    virtual void sweep() = 0;

    // Get the FSR volumes
    virtual std::vector<double> fsr_vol() const = 0;

    // Get the scalar flux
    virtual std::vector<std::vector<double>> scalar_flux() const = 0;

    // Calculate the fission source
    virtual std::vector<double> fission_source(const double keff) const = 0;

    // Set the total source
    virtual void update_source(const std::vector<double>& fissrc) = 0;
};
