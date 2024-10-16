/*
I considered using a linear algebra library like Eigen, but since I want to
better understand the end-to-end functioning of neural networks (and
experiment with CUDA), I decided to implement a matrix class myself.
*/
#pragma once

#include <iostream>
#include <vector>
#include <functional>

template <typename T>
class Matrix {
  private:
    unsigned int rows, cols;
    std::vector<T> data;

    // Helper: get the index of the element at row i, column j
    unsigned int getIndex(unsigned int i, unsigned int j) const;

  public:
    // Basic functionality --------------------------------------------

    // Note that copy/move constructors/assignment operators do not need to be
    // redefined since the default ones will work fine

    // Constructor that initializes the matrix size (but not the values)
    Matrix(unsigned int rows, unsigned int cols);

    // Constructor that is initialized with a 2D vector
    Matrix(const std::vector<std::vector<T>>& values);

    // Constructor that initializes the matrix size and initializes all values
    // to follow a Gaussian distribution
    static Matrix<T> randGaussian(unsigned int rows, unsigned int cols);

    // Constructor that initializes the matrix size and initializes all values
    // to follow a Gaussian distribution with given mean and standard deviation
    static Matrix<T> randGaussian(unsigned int rows, unsigned int cols, T mean, T stdDev);

    T& operator()(unsigned int i, unsigned int j);

    const T& operator()(unsigned int i, unsigned int j) const;

    unsigned int getRows() const;

    unsigned int getCols() const;

    // Prints shape of matrix in (rows, cols) format
    void printShape() const;

    // Prints the entire matrix
    template <typename T1>
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T1>& obj);

    // Matrix math operations -----------------------------------------

    // Matrix operations return a new matrix and do not modify either operands
    Matrix matMul(const Matrix& other) const;
    Matrix matAdd(const Matrix& other) const;
    Matrix elemWiseMul(const Matrix& other) const;

    // Scalar operations modify the current matrix in place
    void scalarAdd(const T& scalar);
    void scalarMul(const T& scalar);

    // Apply a function elementwise to the matrix
    void applyElementwise(const std::function<T(T)>& fn);

    // Returns a new matrix which is the transpose of the current matrix
    Matrix transpose() const;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& obj) {
    // Print matrix entries with brackets and commas
    os << "[" << std::endl;
    for (unsigned int i = 0; i < obj.getRows(); ++i) {
        os << "\t[ ";
        for (unsigned int j = 0; j < obj.getCols(); j++) {
            os << obj(i, j) << ", ";
        }
        os << "]" << std::endl;
    }
    os << "]" << std::endl;
    return os;
}
