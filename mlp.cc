#include <vector>
#include <functional>
#include <algorithm>  // Used for std::shuffle
#include <random>

#include "matrix.h"
#include "mlp.h"

using namespace std;

template <typename T>
MLP<T>::MLP(const vector<unsigned int>& layerSizes, const ActivationFunction<T>& activFn):
    numLayers(layerSizes.size() - 1), layerSizes(layerSizes), weights(numLayers), biases(numLayers), activFn(activFn)
{
    // Initialize weights and biases
    for (unsigned int i = 0; i < numLayers; ++i) {
        unsigned int inputSize = layerSizes[i];
        unsigned int outputSize = layerSizes[i+1];
        weights.emplace_back(Matrix<T>::randGaussian(inputSize, outputSize));
        biases.emplace_back(Matrix<T>::randGaussian(outputSize, 1));
    }
}

template <typename T>
Matrix<T> MLP<T>::forwardPassLayer(const Matrix<T>& input, unsigned int layer) const {
    Matrix<T> result = weights[layer].matMul(input);
    result.scalarAdd(biases[layer]);
    result.applyElementwise(activFn.activation);
    return result;
}

template <typename T>
vector<Matrix<T>> MLP<T>::forwardPass(const Matrix<T>& input) const {
    Matrix<T> result = input;
    vector<Matrix<T>> activations(numLayers + 1);
    for (unsigned int i = 0; i < numLayers; ++i) {
        activations[i] = result;
        result = forwardPassLayer(result, i);
    }
    activations[numLayers] = result;
    return activations;
}

template <typename T>
void MLP<T>::backwardPass(const vector<Matrix<T>>& activations, const Matrix<T>& target, T learningRate) {
    if (activations.size() != numLayers + 1) {
        throw invalid_argument("Number of activations does not match number of layers");
    }

    Matrix<T> lastActivation = activations.back();

    if (lastActivation.getRows() != target.getRows() || lastActivation.getCols() != target.getCols()) {
        throw invalid_argument("Target dimensions do not match output dimensions");
    }

    Matrix<T> error = target.matAdd(lastActivation.matMul(-1));  // Subtract target from output to get error

    // Update weights and biases using backpropagation
    for (unsigned int i = numLayers; i > 0; --i) {
        // Calculate required gradients
        Matrix<T> gradients = error.elemWiseMul(activations[i].applyElementwise(activFn.activationPrime));
        gradients.scalarMul(learningRate);
        Matrix<T> weightGrads = gradients.matMul(activations[i-1].transpose());

        // Update weights and biases
        weights[i-1] = weights[i-1].matAdd(weightGrads);
        biases[i-1] = biases[i-1].matAdd(gradients);

        error = weights[i-1].transpose().matMul(error);
    }
}

template <typename T>
void MLP<T>::train(
    const vector<pair<Matrix<T>, Matrix<T>>>& data,
    T learningRate, unsigned int epochs, bool verbose
) {
    // Validate input
    if (data.size() == 0) {
        throw invalid_argument("Input size cannot be 0");
    } else if (epochs == 0) {
        throw invalid_argument("Number of epochs cannot be 0");
    } else if (learningRate <= 0) {
        throw invalid_argument("Learning rate must be positive");
    }

    // TODO: Add support for batching (will require changes to forwardPass and backwardPass)

    // Train the network
    for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
        if(verbose) { cout << "Starting Epoch " << epoch << endl; }

        // Initialize random number generator and shuffle the input data
        random_device rd;
        mt19937 g(rd());
        shuffle(data.begin(), data.end(), g);

        for (const auto &p : data) {
            vector<Matrix<T>> activations = forwardPass(p.first);
            backwardPass(activations, p.second, learningRate);
        }
    }
}

template <typename T>
Matrix<float> MLP<T>::predict(const Matrix<float>& input) const {
    vector<Matrix<float>> activations = forwardPass(input);
    return activations.back();
}
