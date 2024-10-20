#pragma once

#include <cmath>
#include "activation.h"

template <typename T>
class Relu : public ActivationFunction<T> {
  protected:
    typename ActivationFunction<T>::F getActivFn() const override {
        return [](T x) { return (x > 0) ? x : 0; };
    }

    typename ActivationFunction<T>::F getDerivFn() const override {
        return [](T x) { return (x > 0) ? 1 : 0; };
    }
};
