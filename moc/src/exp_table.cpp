#include "exp_table.hpp"
#include <vector>

ExpTable::ExpTable() {
    double dx = 1.0 / _rdx;
    _exp_table.resize(_n_intervals + 1, {0.0, 0.0});
    double x1 = _min_val;
    double y1 = 1.0 - std::exp(x1);
    for (int i = 0; i < _n_intervals + 1; i++) {
        double x2 = x1 + dx;
        double y2 = 1.0 - std::exp(x2);
        _exp_table[i][0] = (y2 - y1) * _rdx;
        _exp_table[i][1] = y1 - _exp_table[i][0] * x1;
        x1 = x2;
        y1 = y2;
    }
}
