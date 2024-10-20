#include <vector>
#include <functional>
#include <algorithm>  // Used for std::shuffle
#include <random>
#include <memory>
#include <iostream>

#include "matrix.h"
#include "mlp.h"

using namespace std;

template <typename T>
MLP<T>::MLP(const vector<unsigned int>& layerSizes, const ActivationFunction<T>& activFn):
    numLayers(layerSizes.size() - 1), layerSizes(layerSizes), activFn(activFn)
{
    // Initialize weights and biases
    weights.reserve(numLayers);
    biases.reserve(numLayers);
    
    for (unsigned int i = 0; i < numLayers; ++i) {
        unsigned int inputSize = layerSizes[i];
        unsigned int outputSize = layerSizes[i+1];
        weights.emplace_back(make_unique<Matrix<T>>(Matrix<T>::randGaussian(inputSize, outputSize)));
        biases.emplace_back(make_unique<Matrix<T>>(Matrix<T>::randGaussian(1, outputSize)));
    }
}

template <typename T>
Matrix<T> MLP<T>::forwardPassLayer(const Matrix<T>& input, unsigned int layer) const {
    Matrix<T> result = input.matMul(*weights[layer]);
    result = result.matAdd(*biases[layer]);

    // Don't apply activation function to the output layer
    if (layer != numLayers - 1) {
        result.applyElementwise(activFn.getActivationFn());
    }
    // result.applyElementwise(activFn.getActivationFn());
    return result;
}

template <typename T>
typename MLP<T>::MatPtrVec MLP<T>::forwardPass(const Matrix<T>& input) const {
    Matrix<T> result = input;
    typename MLP<T>::MatPtrVec activations(numLayers + 1);
    for (unsigned int i = 0; i < numLayers; ++i) {
        activations[i] = make_unique<Matrix<T>>(result);
        result = forwardPassLayer(result, i);
    }
    activations[numLayers] = make_unique<Matrix<T>>(result);
    return activations;
}

template <typename T>
void MLP<T>::backwardPass(const typename MLP<T>::MatPtrVec& activations, const Matrix<T>& target, T learningRate) {
    if (activations.size() != numLayers + 1) {
        throw invalid_argument("Number of activations does not match number of layers");
    }

    Matrix<T> lastActivation = *activations.back();

    if (lastActivation.getRows() != target.getRows() || lastActivation.getCols() != target.getCols()) {
        throw invalid_argument("Target dimensions do not match output dimensions");
    }

    /*
    Subtract target from output to get error. Note that this is assumed to be the derivative of the loss 
    function with respect to the output at the start. This works for MSE loss (the missing factor of 2
    can be accounted for in the learning rate).

    TODO: Make this more extensible to enable other loss functions
    */
    lastActivation.scalarMul(-1);
    Matrix<T> error = target.matAdd(lastActivation);

    // Update weights and biases using backpropagation
    for (unsigned int i = numLayers; i > 0; --i) {
        // Calculate required gradients
        Matrix<T> activDerivs = *activations[i];
        activDerivs.applyElementwise(activFn.getDerivativeFn());
        Matrix<T> gradients = error.elemWiseMul(activDerivs);
        gradients.scalarMul(learningRate);
        Matrix<T> weightGrads = activations[i-1]->transpose().matMul(gradients);

        // Update weights and biases
        weights[i-1] = make_unique<Matrix<T>>(weights[i-1]->matAdd(weightGrads));
        biases[i-1] = make_unique<Matrix<T>>(
            biases[i-1]->matAdd(gradients.mean(Matrix<T>::ROW))
        );

        error = error.matMul(weights[i-1]->transpose());
    }
}

template <typename T>
void MLP<T>::train(
    vector<pair<Matrix<T>, Matrix<T>>> data,  // Get copy to avoid altering data
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
            typename MLP<T>::MatPtrVec activations = forwardPass(p.first);
            backwardPass(activations, p.second, learningRate);
        }
    }
}

template <typename T>
Matrix<T> MLP<T>::predict(const Matrix<T>& input) const {
    typename MLP<T>::MatPtrVec activations = forwardPass(input);
    return *activations.back();
}

/* 
Explicit instantiation used (rather than .tpp files) to reduce 
recompilation overhead and since I know the types I will be using.
May want to consider using .tpp files in the future.
*/
template class MLP<float>;
template class MLP<double>;
