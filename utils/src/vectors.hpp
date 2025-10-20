#pragma once

#include <vector>

// definitions for common vector instances
using Vector1D = std::vector<double>;
using Vector2D = std::vector<std::vector<double>>;
using Vector3D = std::vector<std::vector<std::vector<double>>>;

// utility functions for vector operations
namespace Vector {

inline void resize(Vector3D& vec, size_t nx, size_t ny, size_t nz, double value = 0.0) {
    vec.resize(nx);
    for (size_t i = 0; i < nx; ++i) {
        vec[i].resize(ny);
        for (size_t j = 0; j < ny; ++j) {
            vec[i][j].resize(nz, value);
        }
    }
}

} // namespace Vector
