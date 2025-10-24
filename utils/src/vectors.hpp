#pragma once

#include <vector>

// definitions for common vector instances
using Vector1D = std::vector<double>;
using Vector2D = std::vector<std::vector<double>>;
using Vector3D = std::vector<std::vector<std::vector<double>>>;
using Vector4D = std::vector<std::vector<std::vector<std::vector<double>>>>;

// utility functions for vector operations
namespace Vector {

inline void resize(Vector1D& vec, size_t N1, double value = 0.0) {
    vec.resize(N1, value);
}

inline void resize(Vector2D& vec, size_t N1, size_t N2, double value = 0.0) {
    vec.resize(N1);
    for (size_t i = 0; i < N1; ++i) {
        vec[i].resize(N2, value);
    }
}

inline void resize(Vector3D& vec, size_t N1, size_t N2, size_t N3, double value = 0.0) {
    vec.resize(N1);
    for (size_t i = 0; i < N1; ++i) {
        vec[i].resize(N2);
        for (size_t j = 0; j < N2; ++j) {
            vec[i][j].resize(N3, value);
        }
    }
}

inline void resize(Vector4D& vec, size_t N1, size_t N2, size_t N3, size_t N4, double value = 0.0) {
    vec.resize(N1);
    for (size_t i = 0; i < N1; ++i) {
        vec[i].resize(N2);
        for (size_t j = 0; j < N2; ++j) {
            vec[i][j].resize(N3);
            for (size_t k = 0; k < N3; ++k) {
                vec[i][j][k].resize(N4, value);
            }
        }
    }
}

} // namespace Vector
