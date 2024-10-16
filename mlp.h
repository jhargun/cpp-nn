#pragma once

#include <vector>
#include <functional>
#include <utility>
#include <memory>

#include "matrix.h"
#include "activations/activation.h"

template <typename T>
class MLP {
    using MatPtrVec = std::vector<std::unique_ptr<Matrix<T>>>;

    // Number of layers in the network
    unsigned int numLayers;

    // Number of neurons in each layer
    std::vector<unsigned int> layerSizes;

    // Weights and biases for each layer
    MatPtrVec weights;
    MatPtrVec biases;

    // Activation function
    const ActivationFunction<T> &activFn;

    // Forward pass helper functions
    Matrix<T> forwardPassLayer(const Matrix<T>& input, unsigned int layer) const;
    MatPtrVec forwardPass(const Matrix<T>& input) const;

    // Backward pass helper functions
    void backwardPass(const MatPtrVec& activations, const Matrix<T>& target, T learningRate);

  public:
    // Constructor (note that layerSizes includes the input layer size)
    MLP(const std::vector<unsigned int>& layerSizes, const ActivationFunction<T>& activFn);

    // Train the network
    // TODO: Add support for mini-batch training
    void train(
        std::vector<std::pair<Matrix<T>, Matrix<T>>> data,
        T learningRate, unsigned int epochs, bool verbose
    );

    // Predict the output of the network
    Matrix<T> predict(const Matrix<T>& input) const;
};
