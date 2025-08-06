#include "cylindrical_solver_base.hpp"

// Explicit template instantiations for the execution spaces we support
template class CylindricalSolverBase<Kokkos::Serial>;

#ifdef KOKKOS_ENABLE_OPENMP
template class CylindricalSolverBase<Kokkos::OpenMP>;
#endif

#ifdef KOKKOS_ENABLE_CUDA
template class CylindricalSolverBase<Kokkos::Cuda>;
#endif
