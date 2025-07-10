#include "argument_parser.hpp"
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

ArgumentParser::ArgumentParser(const std::string& program_name,
                                 const std::string& description)
    : program_name_(program_name), description_(description) {}

void ArgumentParser::add_argument(const std::string& name, const std::string& help) {
    Argument arg;
    arg.name = name;
    arg.help = help;
    arg.required = true;
    arg.is_flag = false;
    positional_args_.push_back(arg);
}

void ArgumentParser::add_option(const std::string& name, const std::string& help,
                const std::string& default_value) {
    Argument arg;
    arg.name = name;
    arg.help = help;
    arg.required = false;
    arg.is_flag = false;
    arg.default_value = default_value;
    optional_args_[name] = arg;
}

// Add a flag (boolean option)
void ArgumentParser::add_flag(const std::string& name, const std::string& help) {
    Argument arg;
    arg.name = name;
    arg.help = help;
    arg.required = false;
    arg.is_flag = true;
    arg.default_value = "false";
    optional_args_[name] = arg;
}

// Add the new overload of add_option
void ArgumentParser::add_option(const std::string& name, const std::string& help,
                              const std::string& default_value,
                              const std::vector<std::string>& valid_values) {
    Argument arg;
    arg.name = name;
    arg.help = help;
    arg.required = false;
    arg.is_flag = false;
    arg.default_value = default_value;
    arg.valid_values = valid_values;
    arg.has_validation = true;

    // Validate default value against valid_values
    if (!default_value.empty() && !valid_values.empty()) {
        if (std::find(valid_values.begin(), valid_values.end(), default_value) == valid_values.end()) {
            std::cerr << "Warning: Default value '" << default_value << "' for option '"
                      << name << "' is not in the list of valid values." << std::endl;
        }
    }

    optional_args_[name] = arg;
}

// Parse command-line arguments
bool ArgumentParser::parse(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc);

    // Check for help flag
    for (const auto& arg : args) {
        if (arg == "-h" || arg == "--help") {
            print_help();
            return false;
        }
    }

    // Process optional arguments and flags
    size_t pos_arg_index = 0;
    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];

	if (arg.substr(0, 9) == "--kokkos-") {
	    continue;
	} else if (arg.substr(0, 1) == "-") {
            // It's an optional argument or flag
            std::string name = arg;
            if (name.size() > 1 && name[1] == '-') {
                // Handle --name format
                name = name.substr(2);
            } else {
                // Handle -name format
                name = name.substr(1);
            }

            auto it = optional_args_.find(name);
            if (it == optional_args_.end()) {
                std::cerr << "Unknown option: " << arg << std::endl;
                print_help();
                return false;
            }

            if (it->second.is_flag) {
                // It's a flag
                it->second.value = "true";
            } else {
                // It's an optional argument that needs a value
                if (i + 1 >= args.size() || args[i+1].substr(0, 1) == "-") {
                    std::cerr << "Option " << arg << " requires a value" << std::endl;
                    print_help();
                    return false;
                }

                std::string value = args[++i];

                // Validate value against valid_values if needed
                if (it->second.has_validation && !it->second.valid_values.empty()) {
                    if (std::find(it->second.valid_values.begin(), it->second.valid_values.end(), value)
                        == it->second.valid_values.end()) {
                        std::cerr << "Error: Invalid value '" << value << "' for option '"
                                 << name << "'." << std::endl;
                        std::cerr << "Valid values are: ";
                        for (size_t j = 0; j < it->second.valid_values.size(); ++j) {
                            std::cerr << "'" << it->second.valid_values[j] << "'";
                            if (j < it->second.valid_values.size() - 1) {
                                std::cerr << ", ";
                            }
                        }
                        std::cerr << std::endl;
                        print_help();
                        return false;
                    }
                }

                it->second.value = value;
            }
        } else {
            // It's a positional argument
            if (pos_arg_index >= positional_args_.size()) {
                std::cerr << "Too many positional arguments" << std::endl;
                print_help();
                return false;
            }

            positional_args_[pos_arg_index].value = arg;
            pos_arg_index++;
        }
    }

    // Check if all required arguments are provided
    if (pos_arg_index < positional_args_.size()) {
        std::cerr << "Not enough positional arguments" << std::endl;
        print_help();
        return false;
    }

    // Set default values for optional arguments not provided
    for (auto& pair : optional_args_) {
        if (pair.second.value.empty()) {
            pair.second.value = pair.second.default_value;
        }
    }

    return true;
}

// Get the value of a positional argument
std::string ArgumentParser::get_positional(size_t index) const {
    if (index < positional_args_.size()) {
        return positional_args_[index].value;
    }
    return "";
}

// Get the value of an optional argument
std::string ArgumentParser::get_option(const std::string& name) const {
    auto it = optional_args_.find(name);
    if (it != optional_args_.end()) {
        return it->second.value;
    }
    return "";
}

// Check if a flag is set
bool ArgumentParser::get_flag(const std::string& name) const {
    auto it = optional_args_.find(name);
    if (it != optional_args_.end() && it->second.is_flag) {
        return it->second.value == "true";
    }
    return false;
}

// Print help message
void ArgumentParser::print_help() const {
    std::cerr << "Usage: " << program_name_;

    for (const auto& arg : positional_args_) {
        std::cerr << " <" << arg.name << ">";
    }

    if (!optional_args_.empty()) {
        std::cerr << " [options]";
    }

    std::cerr << std::endl << std::endl;
    std::cerr << description_ << std::endl << std::endl;

    if (!positional_args_.empty()) {
        std::cerr << "Positional arguments:" << std::endl;
        for (const auto& arg : positional_args_) {
            std::cerr << "  " << arg.name << "\t" << arg.help << std::endl;
        }
        std::cerr << std::endl;
    }

    if (!optional_args_.empty()) {
        std::cerr << "Optional arguments:" << std::endl;
        std::cerr << "  -h, --help\tShow this help message and exit" << std::endl;
        for (const auto& pair : optional_args_) {
            const auto& arg = pair.second;
            std::cerr << "  --" << arg.name;
            if (!arg.is_flag) {
                std::cerr << " VALUE";
            }
            std::cerr << "\t" << arg.help;

            // Show default value if present
            if (!arg.is_flag && !arg.default_value.empty()) {
                std::cerr << " (default: " << arg.default_value << ")";
            }

            // Show valid values if present
            if (arg.has_validation && !arg.valid_values.empty()) {
                std::cerr << " [choices: ";
                for (size_t i = 0; i < arg.valid_values.size(); ++i) {
                    std::cerr << arg.valid_values[i];
                    if (i < arg.valid_values.size() - 1) {
                        std::cerr << ", ";
                    }
                }
                std::cerr << "]";
            }

            std::cerr << std::endl;
        }
    }
}

ArgumentParser ArgumentParser::vera_gpu_moc_parser(const std::string& program_name) {
    ArgumentParser parser(program_name, "VERA GPU Micro-App for eigenvalue calculations");

    // Add required positional arguments
    parser.add_argument("filename", "Input geometry file");
    parser.add_argument("xs_file", "Cross-section data file");

    // Add optional arguments
    parser.add_option("threads", "Number of threads to use", "0");
    parser.add_flag("verbose", "Enable verbose output");
    parser.add_option("sweeper", "Sweeper type (serial, kokkos)", "serial", {"serial", "kokkos"});
    parser.add_option("device", "Device to use (serial, openmp, cuda)", "serial", {"serial", "openmp", "cuda"});
    parser.add_option("max_iter", "Maximum number of iterations", "5000");
    parser.add_option("k_conv_crit", "K-eff convergence criteria threshold", "1e-8");
    parser.add_option("f_conv_crit", "Fission source convergence criteria threshold", "1e-8");

    return parser;
}
