#pragma once
#include <string>
#include <vector>

class MOCBase {
public:
    // Constructor
    MOCBase() = default;

    // Virtual destructor
    virtual ~MOCBase() = default;

    // Run the MOC sweep
    virtual void sweep() = 0;

    // Get the FSR volumes
    virtual const std::vector<double>& fsr_vol() const = 0;

    // Get the scalar flux
    virtual const std::vector<std::vector<double>>& scalar_flux() const = 0;

    // Calculate the fission source
    virtual std::vector<double> fission_source(const double keff) const = 0;

    // Set the total source
    virtual void update_source(const std::vector<double>& fissrc) = 0;
};