#include "th.hpp"
#include <cstdlib>
#include <string>

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

    using Functor = ANTSFunctor<ExecutionSpace>;
    using surface_policy = Kokkos::RangePolicy<ExecutionSpace>;
    using planar_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::planar>;
    using residual_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::surface_residual>;
    using jacobian_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::jacobian_columns>;
    using jacobian_an_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::jacobian_columns_analytic>;
    using partials_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::channel_partials>;
    using copy_gk_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::copy_gk_from_previous_plane>;
    using update_gk_policy = Kokkos::RangePolicy<ExecutionSpace, typename Functor::apply_gk_update>;

    Functor functor(state);
    if (const char* env = std::getenv("SUBCHANNEL_FD_CONSISTENT_X")) functor.fd_consistent_X = (std::string(env) == "1");
    if (const char* env = std::getenv("SUBCHANNEL_DEBUG_CHANNEL")) functor.debug_channel = std::atoi(env);

    const size_t nchan = state.geom->nchannels();
    const size_t nsurf = state.geom->nsurfaces();
    const size_t k = state.surface_plane;
    const size_t k_node = state.node_plane;

    // Copy previous plane solution as starting guess for gk
    if (k_node > 0) {
        Kokkos::parallel_for("TH::gk initial guess", copy_gk_policy(0, nsurf), functor);
    }

    // Sparse Jacobian structure (fixed by the geometry) and a linear solver reused across Newton iterations
    auto jac_row_ptr = state.geom->jacobian_row_ptr_view();
    auto jac_col_idx = state.geom->jacobian_col_idx_view();
    PetscLinearSolver<ExecutionSpace> linear_solver(nsurf, 1.0e-8);
    auto f0 = functor.f0;

    // outer loop for newton iteration convergence
    for (size_t outer_iter = 0; outer_iter < functor.max_outer_iter; ++outer_iter) {

        // accumulate surface sources
        functor.accumulate_surface_sources();

        // PLANAR solve
        Kokkos::parallel_for("TH::planar", planar_policy(0, nchan), functor);

        // calculate the residual vector f0
        Kokkos::parallel_for("TH::solve_surface_mass_flux - calculate residuals f0", residual_policy(0, nsurf), functor);

        // calculate max residual
        double max_res = 0.0;
        Kokkos::parallel_reduce("TH::solve_surface_mass_flux - max residual", surface_policy(0, nsurf),
            KOKKOS_LAMBDA(const size_t ns, double& lmax) {
                // a NaN residual must never look converged: map it to +inf
                const double r = Kokkos::isnan(f0(ns)) ? Kokkos::Experimental::infinity_v<double> : Kokkos::abs(f0(ns));
                if (r > lmax) lmax = r;
            }, Kokkos::Max<double>(max_res));
        std::cout << "Outer Iteration: " << std::setw(3) << outer_iter + 1 << ", Max Residual: " << std::scientific << max_res << std::defaultfloat << std::endl;

        if (max_res < functor.tol) {
            std::cout << "Converged plane " << k << " in " << outer_iter + 1 << " iterations." << std::endl;
            break;
        }

        // Check if max iterations reached
        if (outer_iter == functor.max_outer_iter - 1) {
            std::cout << "WARNING: Plane " << k << " reached max outer iterations (" << functor.max_outer_iter
                      << ") with residual = " << std::scientific << max_res << std::defaultfloat << std::endl;
        }

        // Jacobian: analytic (per-channel partials + chain rule), finite difference, or both for checking
        Kokkos::Profiling::pushRegion("TH::solve_surface_mass_flux - jacobian");
        if (functor.jacobian_mode == 1) {
            Kokkos::parallel_for("TH::jacobian_columns", jacobian_policy(0, nsurf), functor);
        } else {
            Kokkos::parallel_for("TH::channel_partials", partials_policy(0, nchan), functor);
            Kokkos::parallel_for("TH::jacobian_columns_analytic", jacobian_an_policy(0, nsurf), functor);
            if (functor.jacobian_mode == 2) {
                // Compare against the finite-difference Jacobian (same CSR layout)
                auto an = functor.jac_values;
                Kokkos::deep_copy(functor.jac_values_fd, an);
                Kokkos::parallel_for("TH::jacobian_columns", jacobian_policy(0, nsurf), functor); // overwrites jac_values with FD
                auto fd = functor.jac_values;
                functor.jac_values = functor.jac_values_fd;   // analytic goes back into jac_values
                functor.jac_values_fd = fd;
                double max_abs = 0.0, max_rel = 0.0, max_fd = 0.0;
                const size_t nnz = an.extent(0);
                auto a_v = functor.jac_values; auto f_v = functor.jac_values_fd;
                Kokkos::parallel_reduce("TH::jacobian check", surface_policy(0, nnz),
                    KOKKOS_LAMBDA(const size_t q, double& labs, double& lrel, double& lfd) {
                        const double d = Kokkos::abs(a_v(q) - f_v(q));
                        if (d > labs) labs = d;
                        const double fabs_ = Kokkos::abs(f_v(q));
                        if (fabs_ > lfd) lfd = fabs_;
                        if (fabs_ > 1e-6 * 1.0 && d / fabs_ > lrel) lrel = d / fabs_;
                    }, Kokkos::Max<double>(max_abs), Kokkos::Max<double>(max_rel), Kokkos::Max<double>(max_fd));
                std::cout << "  Jacobian check: max|J_an - J_fd| = " << std::scientific << max_abs
                          << " (max|J_fd| = " << max_fd << "), max rel. diff (|J_fd| > 1e-6) = " << max_rel << std::defaultfloat << std::endl;
                if (std::getenv("SUBCHANNEL_JACOBIAN_DUMP")) {
                    auto h_an = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_v);
                    auto h_fd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f_v);
                    auto h_rp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jac_row_ptr);
                    auto h_ci = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jac_col_idx);
                    std::cout << std::scientific;
                    for (size_t row = 0; row < nsurf && row < 16; ++row) {
                        std::cout << "  row " << row << ":";
                        for (int q = h_rp(row); q < h_rp(row + 1); ++q)
                            std::cout << "  [" << h_ci(q) << "] an=" << h_an(q) << " fd=" << h_fd(q);
                        std::cout << std::endl;
                    }
                    std::cout << std::defaultfloat;
                }
            }
        }
        Kokkos::fence();
        Kokkos::Profiling::popRegion();

        // solve the system of equations (overwrites f0 with the solution vector)
        Kokkos::Profiling::pushRegion("TH::solve_surface_mass_flux - solve_linear_system_petsc");
        PetscErrorCode ierr = linear_solver.solve_csr(jac_row_ptr, jac_col_idx, functor.jac_values, functor.f0);
        if (ierr != 0) {
            std::cerr << "PETSc linear solver failed with error code: " << ierr << std::endl;
        }
        Kokkos::Profiling::popRegion();

        // update mass fluxes from solution
        Kokkos::parallel_for("TH::gk update", update_gk_policy(0, nsurf), functor);

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
