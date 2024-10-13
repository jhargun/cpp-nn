#pragma once

#include <cmath>
#include "activation.h"

template <typename T>
class Sigmoid : public ActivationFunction<T> {
  protected:
    T getActivation(T x) const override {
        return 1 / (1 + std::exp(-1 * x));
    }

    T getActivationPrime(T x) const override {
        T s = this->activation(x);
        return s * (1 - s);
    }
};
