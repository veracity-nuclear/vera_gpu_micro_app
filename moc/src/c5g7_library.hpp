#pragma once

#include <string>
#include <vector>

class c5g7_library {
    public:
        // Initialize the library from a file
        c5g7_library(std::string name);
        // Get the number of groups in the library
        int get_num_groups() const;
        // Return the absorption XS for a material
        std::vector<double> abs(int set) const;
        // Return the absorption XS for a material and group
        double abs(int set, int group) const;
        // Return the nu-fission XS for a material
        std::vector<double> nufiss(int set) const;
        // Return the nu-fission XS for a material and group
        double nufiss(int set, int group) const;
        // Return the fission XS for a material
        std::vector<double> fiss(int set) const;
        // Return the fission XS for a material and group
        double fiss(int set, int group) const;
        // Return the chi for a material
        std::vector<double> chi(int set) const;
        // Return the chi for a material and group
        double chi(int set, int group) const;
        // Return the scattering matrix for a material
        std::vector<std::vector<double>> scat(int set) const;
        // Return the scattering row for a material and group
        std::vector<double> scat(int set, int group) const;
        // Return the scattering XS for a material, source group, and destination group
        double scat(int set, int from, int to) const;
        // Return the self-scattering XS for a material
        std::vector<double> self_scat(int set) const;
        // Return the self-scattering XS for a material and group
        double self_scat(int set, int group) const;
        // Return the total XS for a material
        std::vector<double> total(int set) const;
        // Return the total XS for a material and group
        double total(int set, int group) const;
        // Check if a material is fissile
        bool is_fissile(int set) const;
    private:
        std::vector<std::vector<double>> _abs;
        std::vector<std::vector<double>> _nufiss;
        std::vector<std::vector<double>> _fiss;
        std::vector<std::vector<double>> _chi;
        std::vector<std::vector<std::vector<double>>> _scat;
        std::vector<std::vector<double>> _total;
        std::vector<bool> _is_fissile;
};
