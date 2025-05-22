#include "quadrature.hpp"
#include <cmath>

std::vector<double> _gen_cheby(int n) {
    std::vector<double> cheby(n);
    for (int i = 0; i < n; i++) {
        cheby[i] = std::cos(M_PI * (1.0 - double(2 * (2 * n - i) + 1)) / double(4 * n));
    }
    return cheby;
}

std::vector<double> _gen_cheby_weights(int n) {
    std::vector<double> cheby_weights(n);
    for (int i = 0; i < n; i++) {
        cheby_weights[i] = 1.0 / double(n);
    }
    return cheby_weights;
}

std::vector<double> _gen_yamamoto(int n) {
    std::vector<double> yamamoto(n);
    if (n == 1) {
        yamamoto[0] = 0.924274629374;
    } else if (n == 2) {
        yamamoto[0] = 0.372451560620;
        yamamoto[1] = 1.119540153572;
    } else {
        yamamoto[0] = 0.167429147795;
        yamamoto[1] = 0.567715121084;
        yamamoto[2] = 1.202533146789;
    }
    return yamamoto;
}

std::vector<double> _gen_yamamoto_weights(int n) {
    std::vector<double> yamamoto_weights(n);
    if (n == 1) {
        yamamoto_weights[0] = 1.0;
    } else if (n == 2) {
        yamamoto_weights[0] = 0.212854;
        yamamoto_weights[1] = 0.787146;
    } else {
        yamamoto_weights[0] = 0.046233;
        yamamoto_weights[1] = 0.283619;
        yamamoto_weights[2] = 0.670148;
    }
    return yamamoto_weights;
}

Quadrature::Quadrature(int nazi, int npol)
  : _nazi(nazi),
    _npol(npol),
    _azi_angles(_gen_cheby(nazi)),
    _azi_weights(_gen_cheby_weights(nazi)),
    _pol_angles(_gen_yamamoto(npol)),
    _pol_weights(_gen_yamamoto_weights(npol)) {
}