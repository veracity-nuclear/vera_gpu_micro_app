#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>

#include "argument_parser.hpp"
#include "geometry.hpp"
#include "materials.hpp"
#include "solver.hpp"

TEST(SubchannelTest, SMR_Serial) {

    const char* raw_args[] = {
        "exe",
        "../subchannel/data/3DMini.h5",
        "--device", "serial",
        "--no-crossflow"
    };
    char** args = const_cast<char**>(raw_args);
    int argc = 5;

    // Initialize Kokkos
    Kokkos::initialize(argc, args);
    {

        auto parser = ArgumentParser::vera_gpu_subchannel_parser(raw_args[0]);
        parser.parse(argc, args);
        Solver<Kokkos::Serial> solver(parser);
        solver.solve();
        solver.print_state_at_plane(solver.state.surface_plane);

    }
    Kokkos::finalize();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
