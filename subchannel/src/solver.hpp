#pragma once

#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

#include "argument_parser.hpp"
#include "geometry.hpp"
#include "hdf5_kokkos.hpp"
#include "materials.hpp"
#include "state.hpp"
#include "th.hpp"


template <typename ExecutionSpace = Kokkos::DefaultExecutionSpace>
class Solver {
public:

    using MemorySpace = typename ExecutionSpace::memory_space;
    using View1D = Kokkos::View<double*, MemorySpace>;
    using View2D = Kokkos::View<double**, MemorySpace>;
    using View3D = Kokkos::View<double***, MemorySpace>;
    using View4D = Kokkos::View<double****, MemorySpace>;

    Solver(const ArgumentParser& args);
    Solver(
        std::shared_ptr<Geometry<ExecutionSpace>> geometry,
        std::shared_ptr<Water<ExecutionSpace>> fluid,
        View1D inlet_temperature,
        View1D inlet_pressure,
        View1D linear_heat_rate,
        View1D mass_flow_rate
    );
    ~Solver() = default;

    State<ExecutionSpace> state;

    void solve(size_t max_outer_iter = 10, size_t max_inner_iter = 10, bool debug = false);
    void print_state_at_plane(size_t k);

    View2D get_surface_liquid_enthalpies() const { return state.h_l; }
    View2D get_surface_temperatures() const { return state.fluid->T(state.h_l); }
    View2D get_surface_pressures() const { return state.P; }
    View2D get_surface_void_fractions() const { return state.alpha; }
    View2D get_surface_qualities() const { return state.X; }
    View2D get_evaporation_rates() const;
    View2D get_surface_liquid_flow_rates() const { return state.W_l; }
    View2D get_surface_vapor_flow_rates() const { return state.W_v; }

private:
    bool _cf_flag = true;    // flag to turn on/off crossflow solver
};
