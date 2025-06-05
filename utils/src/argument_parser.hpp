#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm> // For std::find

class ArgumentParser {
public:
    struct Argument {
        std::string name;
        std::string help;
        bool required;
        bool is_flag;
        std::string default_value;
        std::string value;
        std::vector<std::string> valid_values; // Added valid values list
        bool has_validation = false;          // Flag to check if validation is needed
    };

    /// Constructor for ArgumentParser
    ArgumentParser(const std::string& program_name, const std::string& description);
    // Add a required positional argument
    void add_argument(const std::string& name, const std::string& help);
    // Add an optional argument with a default value
    void add_option(const std::string& name, const std::string& help,
                const std::string& default_value = "");
    // Add an optional argument with a default value and valid values
    void add_option(const std::string& name, const std::string& help,
                const std::string& default_value,
                const std::vector<std::string>& valid_values);
    // Add a flag (boolean option)
    void add_flag(const std::string& name, const std::string& help);
    // Parse command-line arguments
    bool parse(int argc, char* argv[]);
    // Get the value of a positional argument
    std::string get_positional(size_t index) const;
    // Get the value of an optional argument
    std::string get_option(const std::string& name) const;
    // Check if a flag is set
    bool get_flag(const std::string& name) const;
    // Print the help message
    void print_help() const;
    // Get a vector of arguments compatible with the original EigenSolver constructor
    std::vector<std::string> get_args(const std::string& program_name) const;

private:
    std::string program_name_;
    std::string description_;
    std::vector<Argument> positional_args_;
    std::map<std::string, Argument> optional_args_;
};