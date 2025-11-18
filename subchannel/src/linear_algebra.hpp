#pragma once

#include <iostream>
#include <vector>

#include "vectors.hpp"

inline static int solve_linear_system(int n, Vector2D& A, Vector1D& b) {
    // Forward elimination with partial pivoting
    for (int k = 0; k < n - 1; ++k) {
        // Find pivot
        int pivot_row = k;
        double max_val = std::abs(A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(A[i][k]) > max_val) {
                max_val = std::abs(A[i][k]);
                pivot_row = i;
            }
        }

        // Check for singular matrix
        if (max_val < 1e-14) {
            return k + 1; // Return row index where singularity detected
        }

        // Swap rows if needed
        if (pivot_row != k) {
            std::swap(A[k], A[pivot_row]);
            std::swap(b[k], b[pivot_row]);
        }

        // Eliminate column k
        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            A[i][k] = 0.0;
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Check last diagonal element
    if (std::abs(A[n-1][n-1]) < 1e-14) {
        return n;
    }

    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i][j] * b[j];
        }
        b[i] = sum / A[i][i];
    }

    return 0; // Success
}
