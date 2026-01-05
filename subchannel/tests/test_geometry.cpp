#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <Kokkos_Core.hpp>

#include "geometry.hpp"

TEST(SubchannelTest, GlobalSurfaceIndex) {

    // geometric parameters
    size_t N = 3; // number of subchannels in each direction
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    double length = 1.3e-2; // m, length of axial momentum cell
    size_t naxial = 10; // number of axial nodes to discretize to

    // Create a core map for a single assembly (1x1)
    Kokkos::View<size_t**, Kokkos::Serial> core_map("core_map", 1, 1);
    core_map(0, 0) = 1;

    Geometry<Kokkos::Serial> geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, naxial, core_map);

    EXPECT_EQ(geometry.global_surf_index(0, 0), geometry.boundary);     // Channel 0, W surface
    EXPECT_EQ(geometry.global_surf_index(0, 1), 0);                     // Channel 0, E surface
    EXPECT_EQ(geometry.global_surf_index(0, 2), geometry.boundary);     // Channel 0, N surface
    EXPECT_EQ(geometry.global_surf_index(0, 3), 6);                     // Channel 0, S surface

    EXPECT_EQ(geometry.global_surf_index(1, 0), 0);
    EXPECT_EQ(geometry.global_surf_index(1, 1), 1);
    EXPECT_EQ(geometry.global_surf_index(1, 2), geometry.boundary);
    EXPECT_EQ(geometry.global_surf_index(1, 3), 8);

    EXPECT_EQ(geometry.global_surf_index(2, 0), 1);
    EXPECT_EQ(geometry.global_surf_index(2, 1), geometry.boundary);
    EXPECT_EQ(geometry.global_surf_index(2, 2), geometry.boundary);
    EXPECT_EQ(geometry.global_surf_index(2, 3), 10);

    EXPECT_EQ(geometry.global_surf_index(3, 0), geometry.boundary);
    EXPECT_EQ(geometry.global_surf_index(3, 1), 2);
    EXPECT_EQ(geometry.global_surf_index(3, 2), 6);
    EXPECT_EQ(geometry.global_surf_index(3, 3), 7);

    EXPECT_EQ(geometry.global_surf_index(4, 0), 2);
    EXPECT_EQ(geometry.global_surf_index(4, 1), 3);
    EXPECT_EQ(geometry.global_surf_index(4, 2), 8);
    EXPECT_EQ(geometry.global_surf_index(4, 3), 9);

    EXPECT_EQ(geometry.global_surf_index(5, 0), 3);
    EXPECT_EQ(geometry.global_surf_index(5, 1), geometry.boundary);
    EXPECT_EQ(geometry.global_surf_index(5, 2), 10);
    EXPECT_EQ(geometry.global_surf_index(5, 3), 11);

    EXPECT_EQ(geometry.global_surf_index(6, 0), geometry.boundary);
    EXPECT_EQ(geometry.global_surf_index(6, 1), 4);
    EXPECT_EQ(geometry.global_surf_index(6, 2), 7);
    EXPECT_EQ(geometry.global_surf_index(6, 3), geometry.boundary);

    EXPECT_EQ(geometry.global_surf_index(7, 0), 4);
    EXPECT_EQ(geometry.global_surf_index(7, 1), 5);
    EXPECT_EQ(geometry.global_surf_index(7, 2), 9);
    EXPECT_EQ(geometry.global_surf_index(7, 3), geometry.boundary);

    EXPECT_EQ(geometry.global_surf_index(8, 0), 5);
    EXPECT_EQ(geometry.global_surf_index(8, 1), geometry.boundary);
    EXPECT_EQ(geometry.global_surf_index(8, 2), 11);
    EXPECT_EQ(geometry.global_surf_index(8, 3), geometry.boundary);
}

TEST(SubchannelTest, ToNode_FromNode) {

    // geometric parameters
    size_t N = 3; // number of subchannels in each direction
    double height = 3.81; // m
    double flow_area = 1.436e-4; // m^2
    double hydraulic_diameter = 1.436e-2; // m
    double gap_width = 0.39e-2; // m
    double length = 1.3e-2; // m, length of axial momentum cell
    size_t naxial = 10; // number of axial nodes to discretize to

    // Create a core map for a single assembly (1x1)
    Kokkos::View<size_t**, Kokkos::Serial> core_map("core_map", 1, 1);
    core_map(0, 0) = 1;

    Geometry<Kokkos::Serial> geometry(height, flow_area, hydraulic_diameter, gap_width, length, N, naxial, core_map);

        EXPECT_EQ(geometry.surfaces(0).from_node, 0);
        EXPECT_EQ(geometry.surfaces(0).to_node,   1);

        EXPECT_EQ(geometry.surfaces(1).from_node, 1);
        EXPECT_EQ(geometry.surfaces(1).to_node,   2);

        EXPECT_EQ(geometry.surfaces(2).from_node, 3);
        EXPECT_EQ(geometry.surfaces(2).to_node,   4);

        EXPECT_EQ(geometry.surfaces(3).from_node, 4);
        EXPECT_EQ(geometry.surfaces(3).to_node,   5);

        EXPECT_EQ(geometry.surfaces(4).from_node, 6);
        EXPECT_EQ(geometry.surfaces(4).to_node,   7);

        EXPECT_EQ(geometry.surfaces(5).from_node, 7);
        EXPECT_EQ(geometry.surfaces(5).to_node,   8);

        EXPECT_EQ(geometry.surfaces(6).from_node, 0);
        EXPECT_EQ(geometry.surfaces(6).to_node,   3);

        EXPECT_EQ(geometry.surfaces(7).from_node, 3);
        EXPECT_EQ(geometry.surfaces(7).to_node,   6);

        EXPECT_EQ(geometry.surfaces(8).from_node, 1);
        EXPECT_EQ(geometry.surfaces(8).to_node,   4);

        EXPECT_EQ(geometry.surfaces(9).from_node, 4);
        EXPECT_EQ(geometry.surfaces(9).to_node,   7);

        EXPECT_EQ(geometry.surfaces(10).from_node, 2);
        EXPECT_EQ(geometry.surfaces(10).to_node,   5);

        EXPECT_EQ(geometry.surfaces(11).from_node, 5);
        EXPECT_EQ(geometry.surfaces(11).to_node,   8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
