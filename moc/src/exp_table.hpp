#pragma once

#include <vector>

// Defines an exponential table for fast computation of the exponential function
// for values in the range [-40, 0] with a resolution of 1/1000.
class ExpTable {
    public:
        // Constructor
        ExpTable();
        // Call to retrieve the value of the table
        double expt(const double xval) const;
    private:
        // Exponential table data; (n_intervals + 1, 2)
        std::vector<std::vector<double>> _exp_table;
        // Minimum argument value represented in the table
        double _min_val = -40.0;
        // Maximum argument value represented in the table
        double _max_val = 0.0;
        // Minimum value offset in the table
        int _n_intervals = 40000;
        // Inverse spacing
        double _rdx = _n_intervals / (_max_val - _min_val);
};