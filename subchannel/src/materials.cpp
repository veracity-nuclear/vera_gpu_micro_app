#include "materials.hpp"

template <typename ExecutionSpace>
double Water<ExecutionSpace>::h(double T) const {
    // Simple linear approximation for water enthalpy
    return Cp(0) * (T - 273.15); // J/kg
}

template <typename ExecutionSpace>
typename Water<ExecutionSpace>::DoubleView2D Water<ExecutionSpace>::h(const DoubleView2D& T) const {
    DoubleView2D h_values("h_values", T.extent(0), T.extent(1));
    for (size_t i = 0; i < T.extent(0); ++i) {
        for (size_t k = 0; k < T.extent(1); ++k) {
            h_values(i, k) = h(T(i, k));
        }
    }
    return h_values;
}

template <typename ExecutionSpace>
double Water<ExecutionSpace>::T(double h) const {
    return h / 4220.0 + 273.15; // K, only true because specific heat is constant
}

template <typename ExecutionSpace>
typename Water<ExecutionSpace>::DoubleView2D Water<ExecutionSpace>::T(const DoubleView2D& h) const {
    DoubleView2D T_values("T_values", h.extent(0), h.extent(1));
    for (size_t i = 0; i < h.extent(0); ++i) {
        for (size_t k = 0; k < h.extent(1); ++k) {
            T_values(i, k) = T(h(i, k));
        }
    }
    return T_values;
}

template <typename ExecutionSpace>
double Water<ExecutionSpace>::rho(double h) const {
    // Simple approximation for water density
    return 958.0; // kg/m^3
}

template <typename ExecutionSpace>
typename Water<ExecutionSpace>::DoubleView2D Water<ExecutionSpace>::rho(const DoubleView2D& h) const {
    DoubleView2D rho_values("rho_values", h.extent(0), h.extent(1));
    for (size_t i = 0; i < h.extent(0); ++i) {
        for (size_t k = 0; k < h.extent(1); ++k) {
            rho_values(i, k) = rho(h(i, k));
        }
    }
    return rho_values;
}

template <typename ExecutionSpace>
double Water<ExecutionSpace>::Cp(double h) const {
    // Simple approximation for water specific heat capacity
    return 4220.0; // J/kg-K
}

template <typename ExecutionSpace>
typename Water<ExecutionSpace>::DoubleView2D Water<ExecutionSpace>::Cp(const DoubleView2D& h) const {
    DoubleView2D Cp_values("Cp_values", h.extent(0), h.extent(1));
    for (size_t i = 0; i < h.extent(0); ++i) {
        for (size_t k = 0; k < h.extent(1); ++k) {
            Cp_values(i, k) = Cp(h(i, k));
        }
    }
    return Cp_values;
}

template <typename ExecutionSpace>
double Water<ExecutionSpace>::mu(double h) const {
    // Simple approximation for water viscosity (value for saturated liquid at 7 MPa)
    return 0.001352; // Pa-s
}

template <typename ExecutionSpace>
typename Water<ExecutionSpace>::DoubleView2D Water<ExecutionSpace>::mu(const DoubleView2D& h) const {
    DoubleView2D mu_values("mu_values", h.extent(0), h.extent(1));
    for (size_t i = 0; i < h.extent(0); ++i) {
        for (size_t k = 0; k < h.extent(1); ++k) {
            mu_values(i, k) = mu(h(i, k));
        }
    }
    return mu_values;
}

template <typename ExecutionSpace>
double Water<ExecutionSpace>::k(double h) const {
    // Simple approximation for water thermal conductivity (approximate value for water at 250°C, 7 MPa)
    return 0.6; // W/m-K
}

template <typename ExecutionSpace>
typename Water<ExecutionSpace>::DoubleView2D Water<ExecutionSpace>::k(const DoubleView2D& h) const {
    DoubleView2D k_values("k_values", h.extent(0), h.extent(1));
    for (size_t i = 0; i < h.extent(0); ++i) {
        for (size_t plane = 0; plane < h.extent(1); ++plane) {
            k_values(i, plane) = k(h(i, plane));
        }
    }
    return k_values;
}

// Explicit template instantiations
template class Water<Kokkos::DefaultExecutionSpace>;
template class Water<Kokkos::Serial>;
#if defined(KOKKOS_ENABLE_SERIAL) && !defined(KOKKOS_ENABLE_OPENMP)
template class Water<Kokkos::Serial>;
#endif
