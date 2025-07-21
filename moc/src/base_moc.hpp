#pragma once
#include <string>
#include <vector>

constexpr int RAY_START = 0; // Index for the start of the ray
constexpr int RAY_END = 1;   // Index for the end of the ray

// Defines the angular flux boundary condition for a single face and a single angle
class AngFluxBCFace
{
    public:
        // A 3D vector to hold the angular flux values, indexed by boundary condition, polar angle, and group
        std::vector<std::vector<std::vector<double>>> _angflux;
        // Default constructor
        AngFluxBCFace() = default;
        // Constructor that initializes the angular flux to 0.0 with a specified size
        AngFluxBCFace(int nbc, int npol, int ng) {_resize_angflux(nbc, npol, ng, 0.0);};
        // Constructor that initializes the angular flux to a value with a specified size
        AngFluxBCFace(int nbc, int npol, int ng, double val) {_resize_angflux(nbc, npol, ng, val);};
    private:
        void _resize_angflux(int nbc, int npol, int ng, double val) {
            _angflux.resize(nbc);
            for (size_t i = 0; i < nbc; i++) {
                _angflux[i].resize(npol);
                for (size_t j = 0; j < npol; j++) {
                    _angflux[i][j].resize(ng, val);
                }
            }
        };
};

// Templated version for precision control
template <typename RealType>
class AngFluxBCFaceT
{
    public:
        // A 3D vector to hold the angular flux values, indexed by boundary condition, polar angle, and group
        std::vector<std::vector<std::vector<RealType>>> _angflux;
        // Default constructor
        AngFluxBCFaceT() = default;
        // Constructor that initializes the angular flux to 0.0 with a specified size
        AngFluxBCFaceT(int nbc, int npol, int ng) {_resize_angflux(nbc, npol, ng, static_cast<RealType>(0.0));};
        // Constructor that initializes the angular flux to a value with a specified size
        AngFluxBCFaceT(int nbc, int npol, int ng, RealType val) {_resize_angflux(nbc, npol, ng, val);};
    private:
        void _resize_angflux(int nbc, int npol, int ng, RealType val) {
            _angflux.resize(nbc);
            for (size_t i = 0; i < nbc; i++) {
                _angflux[i].resize(npol);
                for (size_t j = 0; j < npol; j++) {
                    _angflux[i][j].resize(ng, val);
                }
            }
        };
};

// Defines the angular flux boundary condition for a single azimuthal angle
class AngFluxBCAngle
{
    public:
        // A vector of faces, each containing the angular flux for that face
        std::vector<AngFluxBCFace> faces;
        // Default constructor
        AngFluxBCAngle() = default;
        // Constructor that initializes the faces vector with a specified number of faces
        AngFluxBCAngle(int nfaces) {
           faces.resize(nfaces);
        };
};

// Templated version for precision control
template <typename RealType>
class AngFluxBCAngleT
{
    public:
        // A vector of faces, each containing the angular flux for that face
        std::vector<AngFluxBCFaceT<RealType>> faces;
        // Default constructor
        AngFluxBCAngleT() = default;
        // Constructor that initializes the faces vector with a specified number of faces
        AngFluxBCAngleT(int nfaces) {
           faces.resize(nfaces);
        };
};

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
