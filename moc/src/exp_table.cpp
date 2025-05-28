#include "exp_table.hpp"
#include <cmath>
#include <vector>
#include <iostream>

ExpTable::ExpTable() {
    double dx = 1.0 / _rdx;
    _exp_table.resize(_n_intervals + 1, {0.0, 0.0});
    double x1 = _min_val;
    double y1 = 1.0 - std::exp(x1);
    for (int i = 0; i < _n_intervals + 1; ++i) {
        double x2 = x1 + dx;
        double y2 = 1.0 - std::exp(x2);
        _exp_table[i][0] = (y2 - y1) * _rdx;
        _exp_table[i][1] = y1 - _exp_table[i][0] * x1;
        x1 = x2;
        y1 = y2;
    }
}

double ExpTable::expt(const double xval) const {
    int i = std::floor(xval * _rdx) + _n_intervals + 1;
    if (i >= 0 && i < _n_intervals + 1) {
        return _exp_table[i][0] * xval + _exp_table[i][1];
    } else if (xval < -700.0) {
        return 1.0;
    } else {
        return 1.0 - std::exp(xval);
    }
}