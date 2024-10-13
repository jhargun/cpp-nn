#pragma once

#include <vector>
#include <functional>
#include <utility>

#include "matrix.h"
#include "activations/activation.h"

template <typename T>
class MLP {
  private:
    // Number of layers in the network
    unsigned int numLayers;

    // Number of neurons in each layer
    std::vector<unsigned int> layerSizes;

    // Weights and biases for each layer
    std::vector<Matrix<T>> weights;
    std::vector<Matrix<T>> biases;

    // Activation function
    const ActivationFunction<T> &activFn;

    // Forward pass helper functions
    Matrix<T> forwardPassLayer(const Matrix<T>& input, unsigned int layer) const;
    std::vector<Matrix<T>> forwardPass(const Matrix<T>& input) const;

    // Backward pass helper functions
    void backwardPass(const std::vector<Matrix<T>>& activations, const Matrix<T>& target, T learningRate);

  public:
    // Constructor (note that layerSizes includes the input layer size)
    MLP(const std::vector<unsigned int>& layerSizes, const ActivationFunction<T>& activFn);

    // Train the network
    // TODO: Add support for mini-batch training
    void train(
        const std::vector<std::pair<Matrix<T>, Matrix<T>>>& data,
        T learningRate, unsigned int epochs, bool verbose
    );

    // Predict the output of the network
    Matrix<float> predict(const Matrix<float>& input) const;
};
