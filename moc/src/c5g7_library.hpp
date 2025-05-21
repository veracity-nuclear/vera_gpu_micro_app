#pragma once

#include <string>
#include <vector>

class c5g7_library {
    public:
        c5g7_library(std::string name);
        std::vector<double> abs(int set) const;
        std::vector<double> nufiss(int set) const;
        std::vector<double> fiss(int set) const;
        std::vector<double> chi(int set) const;
        std::vector<std::vector<double>> scat(int set) const;
        std::vector<double> scat(int set, int group) const;
        std::vector<double> self_scat(int set) const;
        double self_scat(int set, int group) const;
        std::vector<double> total(int set) const;
    private:
        std::vector<std::vector<double>> _abs;
        std::vector<std::vector<double>> _nufiss;
        std::vector<std::vector<double>> _fiss;
        std::vector<std::vector<double>> _chi;
        std::vector<std::vector<std::vector<double>>> _scat;
        std::vector<std::vector<double>> _total;
};
