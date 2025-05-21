#include "c5g7_library.hpp"
#include <fstream>
#include <sstream>

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

    // Read the file line by line
    std::string line;
    int iset = -1;
    int iline = 0;
    int isetline = -1;
    while (std::getline(input_file, line)) {
        if (iline < 3 || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            iline++;
            continue;
        }
        if (line.substr(0, 1) == "!") {
            iline++;
            continue;
        }
        if (line.substr(0, 7) == "XSMACRO") {
            iset += 1;
            _abs[iset].resize(7);
            _nufiss[iset].resize(7);
            _fiss[iset].resize(7);
            _chi[iset].resize(7);
            _scat[iset].resize(7);
            _total[iset].resize(7);
            isetline = 0;
            iline++;
            continue;
        }
        if (isetline < 7) {
            // Parse the line to get the 4 doubles
            std::stringstream ss(line);
            if (!(ss >> _abs[iset][isetline] >> _nufiss[iset][isetline] >> _fiss[iset][isetline] >> _chi[iset][isetline])) {
                throw std::runtime_error("Failed to parse line: " + line);
            }
            _total[iset][isetline] = _abs[iset][isetline] + _fiss[iset][isetline];
            isetline++;
        }
        else {
            int ig = isetline - 7;
            _scat[iset][ig].resize(7);
            std::stringstream ss(line);
            if (!(ss >> _scat[iset][ig][0] >> _scat[iset][ig][1] >> _scat[iset][ig][2] >> _scat[iset][ig][3] >> _scat[iset][ig][4] >> _scat[iset][ig][5] >> _scat[iset][ig][6])) {
                throw std::runtime_error("Failed to parse line: " + line);
            }
            for (const auto& val : _scat[iset][ig]) {
                _total[iset][ig] += val;
            }
            isetline++;
        }
    }

    input_file.close();
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
