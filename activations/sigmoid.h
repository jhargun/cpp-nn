#pragma once

#include <cmath>
#include "activation.h"

template <typename T>
class Sigmoid : public ActivationFunction<T> {
  protected:
    typename ActivationFunction<T>::F getActivFn() const override {
        return [](T x) { return 1 / (1 + exp(-x)); };
    }

    typename ActivationFunction<T>::F getDerivFn() const override {
        return [](T x) {
            T sig = 1 / (1 + exp(-x));
            return sig * (1 - sig);
        };
    }
};
