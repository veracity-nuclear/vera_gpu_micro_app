#include "argument_parser.hpp"
#include <cstring>
#include <gtest/gtest.h>
#include "petscksp.h"

// Helper function to redirect stderr during tests
class CaptureStderr {
    std::streambuf* old_buffer;
    std::ostringstream captured_stderr;
public:
    CaptureStderr() {
        old_buffer = std::cerr.rdbuf(captured_stderr.rdbuf());
    }

    ~CaptureStderr() {
        std::cerr.rdbuf(old_buffer);
    }

    std::string get_output() const {
        return captured_stderr.str();
    }
};

// Convert string arguments to argc/argv format
void make_args(const std::vector<std::string>& args, int& argc, char**& argv) {
    argc = args.size();
    argv = new char*[argc];

    for (int i = 0; i < argc; i++) {
        argv[i] = new char[args[i].size() + 1];
        std::strcpy(argv[i], args[i].c_str());
    }
}

// Cleanup argv memory
void cleanup_args(int argc, char** argv) {
    for (int i = 0; i < argc; i++) {
        delete[] argv[i];
    }
    delete[] argv;
}

// Test constructor and basic setup
TEST(BasicTest, Constructor) {
    // Not much to test here - just make sure it doesn't crash
    ArgumentParser parser("test_program", "Test description");
}

// Test adding and retrieving positional arguments
TEST(BasicTest, PositionalArguments) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_argument("output", "Output file");

    std::vector<std::string> args = {"test_program", "file1.txt", "file2.txt"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    ASSERT_TRUE(parser.parse(argc, argv));

    std::string input = parser.get_positional(0);
    std::string output = parser.get_positional(1);

    ASSERT_EQ(input, "file1.txt");
    ASSERT_EQ(output, "file2.txt");

    cleanup_args(argc, argv);
}

// Test adding and retrieving optional arguments
TEST(BasicTest, OptionalArguments) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_option("threads", "Number of threads", "1");
    parser.add_option("output", "Output file", "output.txt");

    // Test with provided values
    std::vector<std::string> args = {"test_program", "--threads", "4", "--output", "custom.txt"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    ASSERT_TRUE(parser.parse(argc, argv));

    std::string threads = parser.get_option("threads");
    std::string output = parser.get_option("output");

    ASSERT_EQ(threads, "4");
    ASSERT_EQ(output, "custom.txt");

    cleanup_args(argc, argv);
}

// Test default values for optional arguments
TEST(BasicTest, DefaultValues) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_option("threads", "Number of threads", "1");
    parser.add_option("output", "Output file", "output.txt");
    int argc;
    char** argv;
    // Test with no optional arguments provided
    std::vector<std::string> args = {"test_program"};

    // Test with default values
    std::vector<std::string> args2 = {"test_program"};
    make_args(args2, argc, argv);

    ASSERT_TRUE(parser.parse(argc, argv));

    std::string threads = parser.get_option("threads");
    std::string output = parser.get_option("output");

    ASSERT_EQ(threads, "1");
    ASSERT_EQ(output, "output.txt");

    cleanup_args(argc, argv);
}

// Test option validation
TEST(BasicTest, OptionValidation) {
    ArgumentParser parser("test_program", "Test description");
    std::vector<std::string> valid_levels = {"low", "medium", "high"};
    parser.add_option("level", "Detail level", "medium", valid_levels);

    // Test with valid value
    std::vector<std::string> args = {"test_program", "--level", "high"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    ASSERT_TRUE(parser.parse(argc, argv));

    std::string level = parser.get_option("level");
    ASSERT_EQ(level, "high");

    cleanup_args(argc, argv);

    // Test with invalid value
    std::vector<std::string> args2 = {"test_program", "--level", "ultra"};
    make_args(args2, argc, argv);

    CaptureStderr capture;
    ASSERT_FALSE(parser.parse(argc, argv));
    std::string error_output = capture.get_output();

    ASSERT_NE(error_output.find("Invalid value"), std::string::npos);

    cleanup_args(argc, argv);
}

// Test boolean flags
TEST(BasicTest, Flags) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_flag("verbose", "Verbosity flag");
    parser.add_flag("debug", "Debug flag");

    // Test with flags provided
    std::vector<std::string> args = {"test_program", "--verbose"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    ASSERT_TRUE(parser.parse(argc, argv));

    bool verbose = parser.get_flag("verbose");
    bool debug = parser.get_flag("debug");

    ASSERT_TRUE(verbose);
    ASSERT_FALSE(debug);

    cleanup_args(argc, argv);
}

// Test help flag and output
TEST(BasicTest, Help) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_option("output", "Output file", "out.txt");
    parser.add_flag("verbose", "Enable verbose output");

    std::vector<std::string> args = {"test_program", "--help"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    CaptureStderr capture;

    ASSERT_FALSE(parser.parse(argc, argv)); // Help should return false to stop further processing
    std::string help_output = capture.get_output();
    ASSERT_NE(help_output.find("Usage:"), std::string::npos);
    ASSERT_NE(help_output.find("input"), std::string::npos);
    ASSERT_NE(help_output.find("output"), std::string::npos);
    ASSERT_NE(help_output.find("verbose"), std::string::npos);

    cleanup_args(argc, argv);
}

// Test error cases
TEST(BasicTest, ErrorCases) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_option("output", "Output file", "out.txt");

    // Test missing required argument
    std::vector<std::string> args = {"test_program"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    CaptureStderr capture1;
    ASSERT_FALSE(parser.parse(argc, argv));
    std::string error_output = capture1.get_output();

    ASSERT_NE(error_output.find("Not enough positional arguments"), std::string::npos);

    cleanup_args(argc, argv);

    // Test unknown option
    std::vector<std::string> args2 = {"test_program", "input.txt", "--unknown", "value"};
    make_args(args2, argc, argv);

    CaptureStderr capture2;
    ASSERT_FALSE(parser.parse(argc, argv));
    error_output = capture2.get_output();

    ASSERT_NE(error_output.find("Unknown option"), std::string::npos);

    cleanup_args(argc, argv);

    // Test missing value for option
    std::vector<std::string> args3 = {"test_program", "input.txt", "--output"};
    make_args(args3, argc, argv);

    CaptureStderr capture3;
    ASSERT_FALSE(parser.parse(argc, argv));
    error_output = capture3.get_output();

    ASSERT_NE(error_output.find("requires a value"), std::string::npos);

    cleanup_args(argc, argv);
}

// Test get_args method
TEST(BasicTest, GetArgs) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_argument("output", "Output file");
    parser.add_option("threads", "Number of threads", "1");

    std::vector<std::string> args = {"test_program", "in.txt", "out.txt", "--threads", "4"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    ASSERT_TRUE(parser.parse(argc, argv));

    cleanup_args(argc, argv);
}

TEST(BasicTest, PetscArgs) {
    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");

    std::vector<std::string> args = {"test_program", "in.txt", "-pc_type", "lu"};

    int argc;
    char** argv;
    make_args(args, argc, argv);

    // Ensure argv is null-terminated for c (Petsc) compatibility
    {
        char** argv_with_null = new char*[argc + 1];
        for (int i = 0; i < argc; ++i) argv_with_null[i] = argv[i];
        argv_with_null[argc] = nullptr;
        delete[] argv;
        argv = argv_with_null;
    }

    ASSERT_TRUE(parser.parse(argc, argv));

    {
        PetscInitialize(&argc, &argv, NULL, NULL);

        PC pcTest;
        PetscCallAbort(PETSC_COMM_SELF, PCCreate(PETSC_COMM_SELF, &pcTest));
        PetscCallAbort(PETSC_COMM_SELF, PCSetType(pcTest, PCJACOBI));

        // Sets the options from the command line
        PetscCallAbort(PETSC_COMM_SELF, PCSetFromOptions(pcTest));

        PCType testType;
        PetscCallAbort(PETSC_COMM_SELF, PCGetType(pcTest, &testType));

        ASSERT_STREQ(testType, PCLU) << "PETSc command line argument parsing failed";

        PetscCallAbort(PETSC_COMM_SELF, PCDestroy(&pcTest));
        PetscFinalize();
    }

    cleanup_args(argc, argv);
}

// Test the vera_gpu_moc_parser specifically with ray_sort option
TEST(BasicTest, VeraGpuMocParserRaySortOption) {
    ArgumentParser parser = ArgumentParser::vera_gpu_moc_parser("test_program");
    
    // Test with valid ray_sort values
    std::vector<std::string> args = {"test_program", "input.h5", "xs.h5", "--ray_sort", "long"};
    int argc;
    char** argv;
    make_args(args, argc, argv);
    
    ASSERT_TRUE(parser.parse(argc, argv));
    
    std::string ray_sort = parser.get_option("ray_sort");
    ASSERT_EQ(ray_sort, "long");
    
    cleanup_args(argc, argv);
    
    // Test with default value
    std::vector<std::string> args2 = {"test_program", "input.h5", "xs.h5"};
    make_args(args2, argc, argv);
    
    ASSERT_TRUE(parser.parse(argc, argv));
    
    ray_sort = parser.get_option("ray_sort");
    ASSERT_EQ(ray_sort, "none");
    
    cleanup_args(argc, argv);
    
    // Test with invalid value
    std::vector<std::string> args3 = {"test_program", "input.h5", "xs.h5", "--ray_sort", "invalid"};
    make_args(args3, argc, argv);
    
    CaptureStderr capture;
    ASSERT_FALSE(parser.parse(argc, argv));
    std::string error_output = capture.get_output();
    
    ASSERT_NE(error_output.find("Invalid value"), std::string::npos);
    
    cleanup_args(argc, argv);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}