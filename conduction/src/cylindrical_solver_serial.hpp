#pragma once

#include "cylindrical_solver_base.hpp"

class CylindricalSolverSerial : public CylindricalSolverBase<Kokkos::Serial> {
public:
    using Base = CylindricalSolverBase<Kokkos::Serial>;
    using execution_space = Kokkos::Serial;
    using DoubleView1D = typename Base::DoubleView1D;

    CylindricalSolverSerial(
        std::vector<std::shared_ptr<CylinderNode>> &nodes,
        const std::vector<std::shared_ptr<Solid>> &materials,
        double T_initial = 600.0
    ) : Base(nodes, materials, T_initial) {}

    DoubleView1D solve(
        const DoubleView1D &qdot,
        double T_outer,
        double tolerance = 1e-6,
        size_t max_iterations = 100
    ) override;

protected:
    DoubleView1D internal_Tsolve(
        const DoubleView1D &qdot,
        double T_outer,
        double tolerance,
        size_t max_iterations
    ) override;
};
