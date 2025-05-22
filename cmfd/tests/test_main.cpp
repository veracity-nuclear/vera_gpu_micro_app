#include <gtest/gtest.h>

TEST(BasicTest, AssertTrue) {
    EXPECT_TRUE(true);
}

TEST(BasicTest, AssertFalse) {
    EXPECT_FALSE(false);
}

TEST(BasicTest, AssertEqual) {
    EXPECT_EQ(1, 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
