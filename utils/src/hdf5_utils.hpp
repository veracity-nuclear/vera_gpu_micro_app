#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <highfive/H5File.hpp>

struct FlatHDF5Data {
    std::vector<double> data;
    std::vector<size_t> shape;

    size_t size() const { return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>()); }
    double& operator[](size_t i) { return data[i];}
};

inline FlatHDF5Data read_flat_hdf5_dataset(const std::string& filename, const std::string& dataset_path) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::DataSet dataset = file.getDataSet(dataset_path);
    std::vector<size_t> dims = dataset.getSpace().getDimensions();

    size_t total_size = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    std::vector<double> flat_data(total_size);

    dataset.read_raw(flat_data.data());

    return {flat_data, dims};
}

inline double read_hdf5_scalar(const std::string& filename, const std::string& dataset_path) {
    HighFive::File file(filename, HighFive::File::ReadOnly);
    HighFive::DataSet dataset = file.getDataSet(dataset_path);

    double value;
    dataset.read(value);
    return value;
}

inline FlatHDF5Data operator+(const FlatHDF5Data& lhs, double scalar) {
    FlatHDF5Data result = lhs;
    for (auto& val : result.data) {
        val += scalar;
    }
    return result;
}

inline FlatHDF5Data operator+(double scalar, const FlatHDF5Data& rhs) {
    return rhs + scalar;
}

inline FlatHDF5Data operator-(const FlatHDF5Data& lhs, double scalar) {
    FlatHDF5Data result = lhs;
    for (auto& val : result.data) {
        val -= scalar;
    }
    return result;
}

inline FlatHDF5Data operator*(const FlatHDF5Data& lhs, double scalar) {
    FlatHDF5Data result = lhs;
    for (auto& val : result.data) {
        val *= scalar;
    }
    return result;
}

inline FlatHDF5Data operator*(double scalar, const FlatHDF5Data& rhs) {
    return rhs * scalar;
}

inline FlatHDF5Data operator/(const FlatHDF5Data& lhs, double scalar) {
    FlatHDF5Data result = lhs;
    for (auto& val : result.data) {
        val /= scalar;
    }
    return result;
}
