#include "th.hpp"

/**
 * ANTS Theory refers to
 *
 * Kropaczek, D. J., Salko, R. K. Jr, Hizoum, B., & Collins, B. S. (2023, July).
 * Advanced two-phase subchannel method via non-linear iteration.
 * Nuclear Engineering and Design, 408, 112328.
 *
 * https://doi.org/10.1016/j.nucengdes.2023.112328
 */

template <typename ExecutionSpace>
void TH::solve_surface_mass_flux(State<ExecutionSpace>& state) {
    Kokkos::Profiling::pushRegion("TH::solve_surface_mass_flux");

    const size_t nchan = state.geom->nchannels();
    const size_t nsurf = state.geom->nsurfaces();
    const size_t k = state.surface_plane;
    const size_t k_node = state.node_plane;
    const double tol = 1e-8; // convergence tolerance
    auto num_neighbors = state.geom->num_neighbors_view();

    // Create host mirrors for geometry data
    auto h_num_neighbors = Kokkos::create_mirror_view(num_neighbors);
    Kokkos::deep_copy(h_num_neighbors, num_neighbors);

    // Copy previous plane solution as starting guess for gk
    auto h_gk = Kokkos::create_mirror_view(state.gk);
    Kokkos::deep_copy(h_gk, state.gk);
    if (k_node > 0) {
        for (size_t ns = 0; ns < nsurf; ++ns) {
            h_gk(ns, k_node) = h_gk(ns, k_node - 1);
        }
    }
    // If k_node == 0, gk is already initialized from inlet BC (should be 0)
    Kokkos::deep_copy(state.gk, h_gk);

    // Create surface mirror once (geometry doesn't change)
    auto surfaces = Kokkos::create_mirror_view(state.geom->surface_view());
    Kokkos::deep_copy(surfaces, state.geom->surface_view());

    using Functor = ANTSFunctor<ExecutionSpace>;
    using planar_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::planar>;
    using planar_perturb_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::planar_perturb>;
    using residual_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::surface_residual>;
    using perturbed_residual_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::perturbed_surface_residual>;

    Functor functor(state);

    auto h_f0 = Kokkos::create_mirror_view(functor.f0);
    auto h_gk_pert = Kokkos::create_mirror_view(functor.gk);

    // outer loop for newton iteration convergence
    for (size_t outer_iter = 0; outer_iter < state.max_outer_iter; ++outer_iter) {

        Kokkos::deep_copy(functor.dfdg, 0.0);

        // accumulate surface sources
        functor.accumulate_surf_sources();

        // PLANAR solve
        Kokkos::parallel_for("TH::planar", planar_policy(0, nchan), functor);

        // calculate the residual vector f0
        Kokkos::parallel_for("TH::solve_surface_mass_flux - calculate residuals f0", residual_policy(0, nsurf), functor);

        // calculate max residual
        Kokkos::deep_copy(h_f0, functor.f0);
        double max_res = 0.0;
        for (size_t ns = 0; ns < nsurf; ++ns) {
            max_res = std::max(max_res, std::abs(h_f0(ns)));
        }

        if (max_res < tol) {
            std::cout << "Converged plane " << k << " in " << outer_iter + 1 << " iterations." << std::endl;
            break;
        }

        // Check if max iterations reached
        if (outer_iter == state.max_outer_iter - 1) {
            std::cout << "WARNING: Plane " << k << " reached max outer iterations (" << state.max_outer_iter
                      << ") with residual = " << std::scientific << max_res << std::defaultfloat << std::endl;
        }

        Kokkos::Profiling::pushRegion("TH::solve_surface_mass_flux - perturbation loop");
        for (size_t ns1 = 0; ns1 < nsurf; ++ns1) {
            functor.current_ns1 = ns1;

            Kokkos::deep_copy(h_gk_pert, functor.gk);

            const double gk0 = h_gk(ns1, k_node);

            // perturb the mass flux at surface ns1
            functor.perturb_surface(ns1);

            // PLANAR_PERTURB solve
            Kokkos::parallel_for("TH::planar_perturb", planar_perturb_policy(0, nchan), functor);

            // now assemble f3 and dfdg(:, ns1) on device
            const size_t nneigh = h_num_neighbors(ns1);
            Kokkos::parallel_for("TH::perturbed_residual", perturbed_residual_policy(0, nneigh), functor);

            h_gk_pert(ns1, k_node) = gk0; // restore original value
            Kokkos::deep_copy(functor.gk, h_gk_pert);
        }
        Kokkos::Profiling::popRegion();

        // solve the system of equations (overwrites f0 as solution vector)
        Kokkos::Profiling::pushRegion("TH::solve_surface_mass_flux - solve_linear_system");
        solve_linear_system<ExecutionSpace>(nsurf, functor.dfdg, functor.f0);
        Kokkos::deep_copy(h_f0, functor.f0);
        Kokkos::Profiling::popRegion();

        // update mass fluxes from solution
        std::cout << "Outer Iteration: " << std::setw(3) << outer_iter + 1 << ", Max Residual: " << std::scientific << max_res << std::defaultfloat << std::endl;
        for (size_t ns = 0; ns < nsurf; ++ns) {
            h_gk(ns, k_node) -= h_f0(ns);
        }
        Kokkos::deep_copy(functor.gk, h_gk);

    } // end outer iteration loop

    Kokkos::Profiling::popRegion();
}

// Explicit template instantiations
namespace TH {

template void solve_surface_mass_flux<Kokkos::Serial>(State<Kokkos::Serial>& state);
template void solve_surface_mass_flux<Kokkos::OpenMP>(State<Kokkos::OpenMP>& state);

#ifdef KOKKOS_ENABLE_CUDA
template void solve_surface_mass_flux<Kokkos::Cuda>(State<Kokkos::Cuda>& state);
#endif

}
