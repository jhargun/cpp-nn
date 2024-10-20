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
Matrix<T>::Matrix(const vector<vector<T>>& values): rows(values.size()), cols(values[0].size()), data(rows * cols) {
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
    using fnType = function<T(unsigned int, unsigned int)>;
    fnType getOtherElem = [&other](unsigned int i, unsigned int j) { return other(i, j); };

    if (cols != other.getCols()) {
        throw invalid_argument("Matrix dimensions not compatible for addition");
    } else if (rows != other.getRows()) {
        if (rows == 1) {
            // Swap order so we only have to handle one special case (slightly inefficient)
            return other.matAdd(*this);
        } else if (other.getRows() == 1) {
            // Broadcast the row vector by returning values from the 0th row for all rows
            getOtherElem = [&other](unsigned int i, unsigned int j) { return other(0, j); };
        } else {
            throw invalid_argument("Matrix dimensions do not match (not compatible for addition)");
        }
    }

    Matrix<T> result(rows, cols);
    for(unsigned int i = 0; i < rows; ++i) {
        for(unsigned int j = 0; j < cols; ++j) {
            result(i, j) = (*this)(i, j) + getOtherElem(i, j);
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

template <typename T>
Matrix<T> Matrix<T>::mean(Axis axis) const {
    Matrix<T> result(1,1);
    T sum = 0;

    switch(axis) {
        case ROW:
            result = Matrix<T>(1, cols);
            for (unsigned int j = 0; j < cols; ++j) {
                sum = 0;
                for (unsigned int i = 0; i < rows; ++i) {
                    sum += (*this)(i, j);
                }
                result(0, j) = sum / rows;
            }
            return result;
        case COL:
            result = Matrix<T>(rows, 1);
            for (unsigned int i = 0; i < rows; ++i) {
                sum = 0;
                for (unsigned int j = 0; j < cols; ++j) {
                    sum += (*this)(i, j);
                }
                result(i, 0) = sum / cols;
            }
            return result;
        case ALL:
            sum = 0;
            for (const auto &val : data) {
                sum += val;
            }
            result(0, 0) = sum / (rows * cols);
            return result;
        default:
            throw invalid_argument("Mean along specified axis not implemented");
    }
}

/* 
Explicit instantiation used (rather than .tpp files) to reduce 
recompilation overhead and since I know the types I will be using.
May want to consider using .tpp files in the future.
*/
template class Matrix<float>;
template class Matrix<double>;
