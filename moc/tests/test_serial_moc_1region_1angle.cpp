#include <string>
#include <vector>
#include <filesystem>
#include <gtest/gtest.h>
#include "../src/serial_moc.hpp"

TEST(BasicTest, cart_1region_7g_1a_1p_serial) {
    const std::vector<std::string> args = {"moc_tests", "data/cart_1region_7g_1a_1p_serial.h5", "data/c5g7.xsl"};
    std::cout << "Current directory contents:" << std::endl;
    // List current directory contents
    for (const auto& entry : std::filesystem::directory_iterator(".")) {
        std::cout << "  " << entry.path().string() << std::endl;
    }

    // Check if data directory exists and list its contents
    const std::filesystem::path data_dir = "data";
    if (std::filesystem::exists(data_dir) && std::filesystem::is_directory(data_dir)) {
        std::cout << "Data directory contents:" << std::endl;
        for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
            std::cout << "  " << entry.path().string() << std::endl;
        }
    } else {
        std::cout << "Data directory does not exist!" << std::endl;
    }
    double result = serial_moc_sweep(args);
    EXPECT_NEAR(result, 0.73822796, 1.0e-8);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
