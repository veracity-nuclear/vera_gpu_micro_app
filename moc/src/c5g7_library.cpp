#include "c5g7_library.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

c5g7_library::c5g7_library(std::string name) {
    std::ifstream input_file(name);
    if (!input_file.is_open()) {
        throw std::runtime_error("Could not open file: " + name);
    }

    // Initialize vectors
    _abs.resize(8);
    _nufiss.resize(8);
    _fiss.resize(8);
    _chi.resize(8);
    _scat.resize(8);
    _total.resize(8);
    _is_fissile.resize(8, false);

    // Read the file line by line
    std::string line;
    int iset = -1;
    int iline = 0;
    int isetline = -1;
    while (std::getline(input_file, line)) {
        // The first 3 lines are header material that we don't need right now
        if (iline < 3 || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            iline++;
            continue;
        }
        // If a line starts with '!', it is a comment and we skip it
        if (line.substr(0, 1) == "!") {
            iline++;
            continue;
        }
        // If a line starts with 'XSMACRO', it indicates a new material set
        if (line.substr(0, 7) == "XSMACRO") {
            iset += 1;
            _abs[iset].resize(7, 0.0);
            _nufiss[iset].resize(7, 0.0);
            _fiss[iset].resize(7, 0.0);
            _chi[iset].resize(7, 0.0);
            _scat[iset].resize(7);
            _total[iset].resize(7, 0.0);
            isetline = 0;
            iline++;
            continue;
        }
        // The first 7 lines after 'XSMACRO' contain absorption, nu-fission, fission, and chi data
        if (isetline < 7) {
            // Parse the line to get the 4 doubles
            std::stringstream ss(line);
            if (!(ss >> _abs[iset][isetline] >> _nufiss[iset][isetline] >> _fiss[iset][isetline] >> _chi[iset][isetline])) {
                throw std::runtime_error("Failed to parse line: " + line);
            }
            if (_nufiss[iset][isetline] > 0.0) {
                _is_fissile[iset] = true;
            }
            _total[iset][isetline] += _abs[iset][isetline];
            isetline++;
        }
        // The next 7 lines contain the scattering matrix
        else {
            int ig = isetline - 7;
            _scat[iset][ig].resize(7, 0.0);
            std::stringstream ss(line);
            for (int ig2 = 0; ig2 < 7; ig2++) {
                if (!(ss >> _scat[iset][ig][ig2])) {
                    throw std::runtime_error("Failed to parse line: " + line);
                }
                _total[iset][ig2] += _scat[iset][ig][ig2];
            }
            isetline++;
        }
    }

    input_file.close();
}

int c5g7_library::get_num_groups() const {
    return _total[0].size();
}

std::vector<double> c5g7_library::abs(int set) const {
    return _abs[set];
}

double c5g7_library::abs(int set, int group) const {
    return _abs[set][group];
}

std::vector<double> c5g7_library::nufiss(int set) const {
    return _nufiss[set];
}

double c5g7_library::nufiss(int set, int group) const {
    return _nufiss[set][group];
}

std::vector<double> c5g7_library::fiss(int set) const {
    return _fiss[set];
}

double c5g7_library::fiss(int set, int group) const {
    return _fiss[set][group];
}

std::vector<double> c5g7_library::chi(int set) const {
    return _chi[set];
}

double c5g7_library::chi(int set, int group) const {
    return _chi[set][group];
}

std::vector<std::vector<double>> c5g7_library::scat(int set) const {
    return _scat[set];
}

std::vector<double> c5g7_library::scat(int set, int group) const {
    return _scat[set][group];
}

double c5g7_library::scat(int set, int from, int to) const {
    return _scat[set][from][to];
}

std::vector<double> c5g7_library::self_scat(int set) const {
    std::vector<double> self_scat;
    self_scat.resize(7);
    for (int g = 0; g < 7; g++) {
        self_scat[g] = _scat[set][g][g];
    }
    return self_scat;
}

double c5g7_library::self_scat(int set, int group) const {
    return _scat[set][group][group];
}

std::vector<double> c5g7_library::total(int set) const {
    return _total[set];
}

double c5g7_library::total(int set, int group) const {
    return _total[set][group];
}

bool c5g7_library::is_fissile(int set) const {
    return _is_fissile[set];
}