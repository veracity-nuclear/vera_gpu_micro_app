#pragma once

#include <iostream>
#include <vector>
#include <Kokkos_Core.hpp>

#include "vectors.hpp"

// Templated overload for Kokkos Views with execution space
template<typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
inline static int solve_linear_system(int n, Kokkos::View<double**, ExecutionSpace>& A, Kokkos::View<double*, ExecutionSpace>& b) {
    // Forward elimination with partial pivoting
    for (int k = 0; k < n - 1; ++k) {
        // Find pivot using parallel reduction with custom reducer
        // Need to copy to host for checking singularity and row indices
        auto h_A_col_k = Kokkos::create_mirror_view(Kokkos::subview(A, Kokkos::ALL(), k));
        Kokkos::deep_copy(h_A_col_k, Kokkos::subview(A, Kokkos::ALL(), k));

        int pivot_row = k;
        double max_val = std::abs(h_A_col_k(k));

        // Custom reducer to find both max value and its index
        using MaxLocReducer = Kokkos::MaxLoc<double, int>;
        typename MaxLocReducer::value_type result;
        result.val = max_val;
        result.loc = k;

        Kokkos::parallel_reduce("find_pivot", Kokkos::RangePolicy<ExecutionSpace>(k + 1, n),
            KOKKOS_LAMBDA(const int i, typename MaxLocReducer::value_type& update) {
                double abs_val = Kokkos::abs(A(i, k));
                if (abs_val > update.val) {
                    update.val = abs_val;
                    update.loc = i;
                }
            },
            MaxLocReducer(result));

        max_val = result.val;
        pivot_row = result.loc;

        // Check for singular matrix
        if (max_val < 1e-14) {
            return k + 1; // Return row index where singularity detected
        }

        // Swap rows if needed
        if (pivot_row != k) {
            Kokkos::parallel_for("swap_rows", Kokkos::RangePolicy<ExecutionSpace>(0, n),
                KOKKOS_LAMBDA(const int j) {
                    double tmp = A(k, j);
                    A(k, j) = A(pivot_row, j);
                    A(pivot_row, j) = tmp;
                });
            Kokkos::fence();

            // Also swap b vector elements
            auto h_b_k = Kokkos::create_mirror_view(Kokkos::subview(b, k));
            auto h_b_pivot = Kokkos::create_mirror_view(Kokkos::subview(b, pivot_row));
            Kokkos::deep_copy(h_b_k, Kokkos::subview(b, k));
            Kokkos::deep_copy(h_b_pivot, Kokkos::subview(b, pivot_row));
            double tmp = h_b_k();
            h_b_k() = h_b_pivot();
            h_b_pivot() = tmp;
            Kokkos::deep_copy(Kokkos::subview(b, k), h_b_k);
            Kokkos::deep_copy(Kokkos::subview(b, pivot_row), h_b_pivot);
        }

        // Eliminate column k
        Kokkos::parallel_for("eliminate", Kokkos::RangePolicy<ExecutionSpace>(k + 1, n),
            KOKKOS_LAMBDA(const int i) {
                double factor = A(i, k) / A(k, k);
                A(i, k) = 0.0;
                for (int j = k + 1; j < n; ++j) {
                    A(i, j) -= factor * A(k, j);
                }
                b(i) -= factor * b(k);
            });
        Kokkos::fence();
    }

    // Check last diagonal element - need to copy to host
    auto h_A_nn = Kokkos::create_mirror_view(Kokkos::subview(A, n-1, n-1));
    Kokkos::deep_copy(h_A_nn, Kokkos::subview(A, n-1, n-1));
    if (Kokkos::abs(h_A_nn()) < 1e-14) {
        return n;
    }

    // Back substitution (sequential outer loop, but can parallelize inner operations)
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        Kokkos::parallel_reduce("back_substitution", Kokkos::RangePolicy<ExecutionSpace>(i + 1, n),
            KOKKOS_LAMBDA(const int j, double& lsum) {
                lsum += A(i, j) * b(j);
            },
            sum);

        // Update b(i) - need to do this on host then copy back
        auto h_A_ii = Kokkos::create_mirror_view(Kokkos::subview(A, i, i));
        auto h_b_i = Kokkos::create_mirror_view(Kokkos::subview(b, i));
        Kokkos::deep_copy(h_A_ii, Kokkos::subview(A, i, i));
        Kokkos::deep_copy(h_b_i, Kokkos::subview(b, i));
        h_b_i() = (h_b_i() - sum) / h_A_ii();
        Kokkos::deep_copy(Kokkos::subview(b, i), h_b_i);
    }

    return 0; // Success
}

// Original overload for std::vector (keeping in case need to use with vectors)
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
