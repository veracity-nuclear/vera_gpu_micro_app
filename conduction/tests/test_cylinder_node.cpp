#include <gtest/gtest.h>
#include "cylinder_node.hpp"

TEST(CylinderNodeTest, ConstructorValid) {
    EXPECT_NO_THROW(CylinderNode(1.0, 0.01, 0.02)); // Cylindrical node with valid parameters
    EXPECT_NO_THROW(CylinderNode(1.0, 0.01, 0.02)); // Cylindrical shell node with valid parameters
}

TEST(CylinderNodeTest, ConstructorInvalidRadius) {
    EXPECT_THROW(CylinderNode(-1.0, 0.01, 0.02), std::invalid_argument); // Negative height
    EXPECT_THROW(CylinderNode(1.0, -0.01, 0.02), std::invalid_argument); // Negative inner radius
    EXPECT_THROW(CylinderNode(1.0, 0.01, -0.02), std::invalid_argument); // Negative outer radius
    EXPECT_THROW(CylinderNode(1.0, 0.02, 0.01), std::invalid_argument); // Inner radius greater than outer radius
}

TEST(CylinderNodeTest, Getters) {
    CylinderNode cyl_node(1.0, 0.01, 0.02);
    EXPECT_DOUBLE_EQ(cyl_node.get_inner_radius(), 0.01);
    EXPECT_DOUBLE_EQ(cyl_node.get_outer_radius(), 0.02);
    EXPECT_DOUBLE_EQ(cyl_node.get_height(), 1.0);
}

TEST(CylinderNodeTest, AreaAndVolume) {
    CylinderNode cyl_node(1.0, 1.0, 2.0);
    EXPECT_DOUBLE_EQ(cyl_node.get_inner_area(), 2.0 * PI); // 2 * PI * r_in * h
    EXPECT_DOUBLE_EQ(cyl_node.get_outer_area(), 4.0 * PI); // 2 * PI * r_out * h
    EXPECT_DOUBLE_EQ(cyl_node.get_volume(), 3.0 * PI); // PI * h * (r_out^2 - r_in^2)
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
