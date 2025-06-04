#include "argument_parser.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <cstring>
#include <gtest/gtest.h>

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
    std::cout << "Testing constructor... ";

    ArgumentParser parser("test_program", "Test description");

    // Not much to test here - just make sure it doesn't crash
    std::cout << "PASSED\n";
}

// Test adding and retrieving positional arguments
TEST(BasicTest, PositionalArguments) {
    std::cout << "Testing positional arguments... ";

    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_argument("output", "Output file");

    std::vector<std::string> args = {"test_program", "file1.txt", "file2.txt"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    bool success = parser.parse(argc, argv);
    assert(success);

    std::string input = parser.get_positional(0);
    std::string output = parser.get_positional(1);

    assert(input == "file1.txt");
    assert(output == "file2.txt");

    cleanup_args(argc, argv);
    std::cout << "PASSED\n";
}

// Test adding and retrieving optional arguments
TEST(BasicTest, OptionalArguments) {
    std::cout << "Testing optional arguments... ";

    ArgumentParser parser("test_program", "Test description");
    parser.add_option("threads", "Number of threads", "1");
    parser.add_option("output", "Output file", "output.txt");

    // Test with provided values
    std::vector<std::string> args = {"test_program", "--threads", "4", "--output", "custom.txt"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    bool success = parser.parse(argc, argv);
    assert(success);

    std::string threads = parser.get_option("threads");
    std::string output = parser.get_option("output");

    assert(threads == "4");
    assert(output == "custom.txt");

    cleanup_args(argc, argv);

    // Test with default values
    std::vector<std::string> args2 = {"test_program"};
    make_args(args2, argc, argv);

    success = parser.parse(argc, argv);
    assert(success);

    threads = parser.get_option("threads");
    output = parser.get_option("output");

    assert(threads == "1");
    assert(output == "output.txt");

    cleanup_args(argc, argv);
    std::cout << "PASSED\n";
}

// Test option validation
TEST(BasicTest, OptionValidation) {
    std::cout << "Testing option validation... ";

    ArgumentParser parser("test_program", "Test description");
    std::vector<std::string> valid_levels = {"low", "medium", "high"};
    parser.add_option("level", "Detail level", "medium", valid_levels);

    // Test with valid value
    std::vector<std::string> args = {"test_program", "--level", "high"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    bool success = parser.parse(argc, argv);
    assert(success);

    std::string level = parser.get_option("level");
    assert(level == "high");

    cleanup_args(argc, argv);

    // Test with invalid value
    std::vector<std::string> args2 = {"test_program", "--level", "ultra"};
    make_args(args2, argc, argv);

    CaptureStderr capture;
    success = parser.parse(argc, argv);
    std::string error_output = capture.get_output();

    assert(!success);
    assert(error_output.find("Invalid value") != std::string::npos);

    cleanup_args(argc, argv);
    std::cout << "PASSED\n";
}

// Test boolean flags
TEST(BasicTest, Flags) {
    std::cout << "Testing boolean flags... ";

    ArgumentParser parser("test_program", "Test description");
    parser.add_flag("verbose", "Verbosity flag");
    parser.add_flag("debug", "Debug flag");

    // Test with flags provided
    std::vector<std::string> args = {"test_program", "--verbose"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    bool success = parser.parse(argc, argv);
    assert(success);

    bool verbose = parser.get_flag("verbose");
    bool debug = parser.get_flag("debug");

    assert(verbose == true);
    assert(debug == false);

    cleanup_args(argc, argv);
    std::cout << "PASSED\n";
}

// Test help flag and output
TEST(BasicTest, Help) {
    std::cout << "Testing help output... ";

    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_option("output", "Output file", "out.txt");
    parser.add_flag("verbose", "Enable verbose output");

    std::vector<std::string> args = {"test_program", "--help"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    CaptureStderr capture;
    bool success = parser.parse(argc, argv);
    std::string help_output = capture.get_output();

    assert(!success); // Help should return false to stop further processing
    assert(help_output.find("Usage:") != std::string::npos);
    assert(help_output.find("input") != std::string::npos);
    assert(help_output.find("output") != std::string::npos);
    assert(help_output.find("verbose") != std::string::npos);

    cleanup_args(argc, argv);
    std::cout << "PASSED\n";
}

// Test error cases
TEST(BasicTest, ErrorCases) {
    std::cout << "Testing error cases... ";

    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_option("output", "Output file", "out.txt");

    // Test missing required argument
    std::vector<std::string> args = {"test_program"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    CaptureStderr capture1;
    bool success = parser.parse(argc, argv);
    std::string error_output = capture1.get_output();

    assert(!success);
    assert(error_output.find("Not enough positional arguments") != std::string::npos);

    cleanup_args(argc, argv);

    // Test unknown option
    std::vector<std::string> args2 = {"test_program", "input.txt", "--unknown", "value"};
    make_args(args2, argc, argv);

    CaptureStderr capture2;
    success = parser.parse(argc, argv);
    error_output = capture2.get_output();

    assert(!success);
    assert(error_output.find("Unknown option") != std::string::npos);

    cleanup_args(argc, argv);

    // Test missing value for option
    std::vector<std::string> args3 = {"test_program", "input.txt", "--output"};
    make_args(args3, argc, argv);

    CaptureStderr capture3;
    success = parser.parse(argc, argv);
    error_output = capture3.get_output();

    assert(!success);
    assert(error_output.find("requires a value") != std::string::npos);

    cleanup_args(argc, argv);
    std::cout << "PASSED\n";
}

// Test get_args method
TEST(BasicTest, GetArgs) {
    std::cout << "Testing get_args method... ";

    ArgumentParser parser("test_program", "Test description");
    parser.add_argument("input", "Input file");
    parser.add_argument("output", "Output file");
    parser.add_option("threads", "Number of threads", "1");

    std::vector<std::string> args = {"test_program", "in.txt", "out.txt", "--threads", "4"};
    int argc;
    char** argv;
    make_args(args, argc, argv);

    bool success = parser.parse(argc, argv);
    assert(success);

    std::vector<std::string> solver_args = parser.get_args("test_program");

    assert(solver_args.size() == 3); // program_name + 2 positional args
    assert(solver_args[0] == "test_program");
    assert(solver_args[1] == "in.txt");
    assert(solver_args[2] == "out.txt");

    cleanup_args(argc, argv);
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "Running ArgumentParser Unit Tests\n";

    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}