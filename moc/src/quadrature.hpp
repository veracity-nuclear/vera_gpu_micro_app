#pragma once

#include <vector>

class Quadrature{
    public:
        Quadrature(int nazi, int npol);
        int nazi() const { return _nazi; }
        int npol() const { return _npol; }
        std::vector<double> azi_angles() const { return _azi_angles; }
        double azi_angle(int i) const { return _azi_angles[i]; }
        std::vector<double> pol_angles() const { return _pol_angles; }
        double pol_angle(int i) const { return _pol_angles[i]; }
        std::vector<double> azi_weights() const { return _azi_weights; }
        double azi_weight(int i) const { return _azi_weights[i]; }
        std::vector<double> pol_weights() const { return _pol_weights; }
        double pol_weight(int i) const { return _pol_weights[i]; }
        int reflect(int angle, int face) const;
    private:
        int _nazi;
        int _npol;
        std::vector<double> _azi_angles;
        std::vector<double> _azi_weights;
        std::vector<double> _pol_angles;
        std::vector<double> _pol_weights;
};