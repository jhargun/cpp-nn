#include <vector>
#include <iostream>
#include <cmath>
#include <random>

#include "matrix.h"

using namespace std;

template <typename T>
unsigned int Matrix<T>::getIndex(unsigned int i, unsigned int j) const {
    return i * cols + j;
}

template <typename T>
Matrix<T>::Matrix(unsigned int rows, unsigned int cols): rows(rows), cols(cols) {
    if (rows <= 0 || cols <= 0) {
        throw invalid_argument("Invalid matrix size");
    }

    // Resize here rather than defining size in the member initializer list
    // lets us ensure that the above error checking is done first
    data.resize(rows * cols);
}

template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>>& values): rows(values.size()), cols(values[0].size()), data(rows * cols) {
    // Check that every row has the same number of columns
    for (const auto &row : values) {
        if (row.size() != cols) {
            throw invalid_argument("All rows must have the same number of columns");
        }
    }

    // Copy the data over
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            (*this)(i, j) = values[i][j];
        }
    }
}


template <typename T>
Matrix<T> Matrix<T>::randGaussian(unsigned int rows, unsigned int cols) {
    // Calculate parameters for the generated data
    T mean{0};
    T std{1 / sqrt(rows * cols)};

    return randGaussian(rows, cols, mean, std);
}

template <typename T>
Matrix<T> Matrix<T>::randGaussian(unsigned int rows, unsigned int cols, T mean, T stdDev) {
    Matrix<T> result(rows, cols);

    // Create a random number generator
    random_device rd;  // Get random seed
    mt19937 gen(rd());  // Seed generator
    normal_distribution<T> dist(mean, stdDev);  // Define the normal distribution

    for (auto &val : result.data) {
        val = dist(gen);
    }
    return result;
}

template <typename T>
T& Matrix<T>::operator()(unsigned int i, unsigned int j) {
    return data[getIndex(i,j)];
}

template <typename T>
const T& Matrix<T>::operator()(unsigned int i, unsigned int j) const {
    return data[getIndex(i,j)];
}

template <typename T>
unsigned int Matrix<T>::getRows() const { return rows; }

template <typename T>
unsigned int Matrix<T>::getCols() const { return cols; }

template <typename T>
void Matrix<T>::printShape() const {
    cout << "(" << rows << ", " << cols << ")" << endl;
}

template <typename T>
ostream& operator<<(ostream& os, const Matrix<T>& obj) {
    // Print matrix entries with brackets and commas
    os << "[" << endl;
    for (unsigned int i = 0; i < obj.getRows(); ++i) {
        os << "\t[ ";
        for (unsigned int j = 0; j < obj.getCols(); j++) {
            os << obj(i, j) << ", ";
        }
        os << "]" << endl;
    }
    os << "]" << endl;
    return os;
}

template <typename T>
Matrix<T> Matrix<T>::matMul(const Matrix<T>& other) const {
    if (cols != other.getRows()) {
        throw invalid_argument("Matrix dimensions not compatible for multiplication");
    }

    Matrix<T> result(rows, other.getCols());

    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < other.getCols(); ++j) {
            T sum = 0;
            for (unsigned int k = 0; k < cols; ++k) {
                // TODO: If T is limited to float/double, can use std::fma
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::matAdd(const Matrix<T>& other) const {
    if (rows != other.getRows() || cols != other.getCols()) {
        throw invalid_argument("Matrix dimensions not compatible for addition");
    }

    Matrix<T> result(rows, cols);
    for(unsigned int i = 0; i < rows; ++i) {
        for(unsigned int j = 0; j < cols; ++j) {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }

    return result;
}

template <typename T>
Matrix<T> Matrix<T>::elemWiseMul(const Matrix<T>& other) const {
    if (rows != other.getRows() || cols != other.getCols()) {
        throw invalid_argument(
            "Matrix dimensions do not match (not compatible for elementwise multiplication)"
        );
    }

    Matrix<T> result(rows, cols);
    for(unsigned int i = 0; i < rows; ++i) {
        for(unsigned int j = 0; j < cols; ++j) {
            result(i, j) = (*this)(i, j) * other(i, j);
        }
    }

    return result;
}

template <typename T>
void Matrix<T>::scalarMul(const T& scalar) {
    for (auto &val : data) { val *= scalar; }
}

template <typename T>
void Matrix<T>::scalarAdd(const T& scalar) {
    for (auto &val : data) { val += scalar; }
}

template <typename T>
void Matrix<T>::applyElementwise(const function<T(T)>& fn) {
    for (auto &val : data) { val = fn(val); }
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    // Create a new matrix with the dimensions swapped
    Matrix<T> result(cols, rows);

    // Copy the data over
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }

    return result;
}

template class Matrix<float>;
template class Matrix<double>;
