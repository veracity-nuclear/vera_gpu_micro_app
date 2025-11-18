#include "materials.hpp"

double Water::h(double T) const {
    // Simple linear approximation for water enthalpy
    return Cp(0) * (T - 273.15); // J/kg
}

Water::DoubleView2D Water::h(const DoubleView2D& T) const {
    DoubleView2D h_values("h_values", T.extent(0), T.extent(1));
    auto T_copy = T;
    auto this_ptr = this;
    Kokkos::parallel_for("Water::h", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {T.extent(0), T.extent(1)}),
        KOKKOS_LAMBDA(const size_t i, const size_t k) {
            h_values(i, k) = this_ptr->h(T_copy(i, k));
        });
    return h_values;
}

double Water::T(double h) const {
    return h / 4220.0 + 273.15; // K, only true because specific heat is constant
}

Water::DoubleView2D Water::T(const DoubleView2D& h) const {
    DoubleView2D T_values("T_values", h.extent(0), h.extent(1));
    auto h_copy = h;
    auto this_ptr = this;
    Kokkos::parallel_for("Water::T", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h.extent(0), h.extent(1)}),
        KOKKOS_LAMBDA(const size_t i, const size_t k) {
            T_values(i, k) = this_ptr->T(h_copy(i, k));
        });
    return T_values;
}

double Water::rho(double h) const {
    // Simple approximation for water density
    return 958.0; // kg/m^3
}

Water::DoubleView2D Water::rho(const DoubleView2D& h) const {
    DoubleView2D rho_values("rho_values", h.extent(0), h.extent(1));
    auto h_copy = h;
    auto this_ptr = this;
    Kokkos::parallel_for("Water::rho", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h.extent(0), h.extent(1)}),
        KOKKOS_LAMBDA(const size_t i, const size_t k) {
            rho_values(i, k) = this_ptr->rho(h_copy(i, k));
        });
    return rho_values;
}

double Water::Cp(double h) const {
    // Simple approximation for water specific heat capacity
    return 4220.0; // J/kg-K
}

Water::DoubleView2D Water::Cp(const DoubleView2D& h) const {
    DoubleView2D Cp_values("Cp_values", h.extent(0), h.extent(1));
    auto h_copy = h;
    auto this_ptr = this;
    Kokkos::parallel_for("Water::Cp", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h.extent(0), h.extent(1)}),
        KOKKOS_LAMBDA(const size_t i, const size_t k) {
            Cp_values(i, k) = this_ptr->Cp(h_copy(i, k));
        });
    return Cp_values;
}

double Water::mu(double h) const {
    // Simple approximation for water viscosity (value for saturated liquid at 7 MPa)
    return 0.001352; // Pa-s
}

Water::DoubleView2D Water::mu(const DoubleView2D& h) const {
    DoubleView2D mu_values("mu_values", h.extent(0), h.extent(1));
    auto h_copy = h;
    auto this_ptr = this;
    Kokkos::parallel_for("Water::mu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h.extent(0), h.extent(1)}),
        KOKKOS_LAMBDA(const size_t i, const size_t k) {
            mu_values(i, k) = this_ptr->mu(h_copy(i, k));
        });
    return mu_values;
}

double Water::k(double h) const {
    // Simple approximation for water thermal conductivity (approximate value for water at 250°C, 7 MPa)
    return 0.6; // W/m-K
}

Water::DoubleView2D Water::k(const DoubleView2D& h) const {
    DoubleView2D k_values("k_values", h.extent(0), h.extent(1));
    auto h_copy = h;
    auto this_ptr = this;
    Kokkos::parallel_for("Water::k", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {h.extent(0), h.extent(1)}),
        KOKKOS_LAMBDA(const size_t i, const size_t plane) {
            k_values(i, plane) = this_ptr->k(h_copy(i, plane));
        });
    return k_values;
}
