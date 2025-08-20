#ifndef ANTS_SUBCHANNEL_IO_UTILS_HPP
#define ANTS_SUBCHANNEL_IO_UTILS_HPP

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace ants {
namespace subchannel {

/**
 * @brief Utilities for formatted output and file I/O
 */
class IOUtils {
public:
    /**
     * @brief Print a formatted table header
     */
    static void printTableHeader(const std::vector<std::string>& headers,
                                const std::vector<int>& widths = {});

    /**
     * @brief Print a separator line
     */
    static void printSeparator(int total_width = 70, char sep_char = '-');

    /**
     * @brief Print formatted results in CSV style
     */
    static void printCSVLine(const std::vector<double>& values,
                            const std::vector<std::string>& labels = {},
                            int precision = 6);

    /**
     * @brief Print a banner with title
     */
    static void printBanner(const std::string& title, char border_char = '=');

    /**
     * @brief Format scientific notation
     */
    static std::string formatScientific(double value, int precision = 3);

    /**
     * @brief Format engineering notation (powers of 1000)
     */
    static std::string formatEngineering(double value, int precision = 3);

private:
    IOUtils() = default; // Static utility class
};

// Inline implementations for header-only usage
inline void IOUtils::printTableHeader(const std::vector<std::string>& headers,
                                     const std::vector<int>& widths) {
    if (headers.empty()) return;

    // Use default width if not specified
    std::vector<int> col_widths = widths;
    if (col_widths.empty()) {
        col_widths.resize(headers.size(), 12);
    }

    // Print headers
    for (size_t i = 0; i < headers.size(); ++i) {
        std::cout << std::setw(col_widths[i]) << headers[i];
        if (i < headers.size() - 1) std::cout << " | ";
    }
    std::cout << std::endl;

    // Print separator
    int total_width = 0;
    for (size_t i = 0; i < col_widths.size(); ++i) {
        total_width += col_widths[i];
        if (i < col_widths.size() - 1) total_width += 3; // " | "
    }
    printSeparator(total_width);
}

inline void IOUtils::printSeparator(int total_width, char sep_char) {
    std::cout << std::string(total_width, sep_char) << std::endl;
}

inline void IOUtils::printCSVLine(const std::vector<double>& values,
                                 const std::vector<std::string>& labels,
                                 int precision) {
    std::cout << std::fixed << std::setprecision(precision);

    for (size_t i = 0; i < values.size(); ++i) {
        if (!labels.empty() && i < labels.size()) {
            std::cout << labels[i] << ": ";
        }
        std::cout << values[i];
        if (i < values.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

inline void IOUtils::printBanner(const std::string& title, char border_char) {
    int width = std::max(50, static_cast<int>(title.length()) + 4);
    std::cout << std::string(width, border_char) << std::endl;

    // Center the title
    int padding = (width - title.length()) / 2;
    std::cout << std::string(padding, ' ') << title << std::endl;
    std::cout << std::string(width, border_char) << std::endl;
}

inline std::string IOUtils::formatScientific(double value, int precision) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(precision) << value;
    return oss.str();
}

inline std::string IOUtils::formatEngineering(double value, int precision) {
    if (value == 0.0) return "0.0";

    int exponent = static_cast<int>(std::floor(std::log10(std::abs(value))));
    int eng_exp = (exponent / 3) * 3;
    double mantissa = value / std::pow(10.0, eng_exp);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << mantissa;
    if (eng_exp != 0) {
        oss << "e" << std::showpos << eng_exp;
    }
    return oss.str();
}

} // namespace subchannel
} // namespace ants

#endif // ANTS_SUBCHANNEL_IO_UTILS_HPP
