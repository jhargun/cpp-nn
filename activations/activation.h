/*
ActivationFunction is an abstract class that defines the interface for 
activation functions. Activation functions should inherit from this class.
*/
#pragma once

#include <functional>

template <typename T>
class ActivationFunction {
  protected:
    using F = std::function<T(T)>;
    virtual F getActivFn() const = 0;
    virtual F getDerivFn() const = 0;
  public:
    virtual ~ActivationFunction() = default;

    // These public functions just call the protected virtual functions above (NVI idiom)
    F getActivationFn() const { return getActivFn(); }
    F getDerivativeFn() const { return getDerivFn(); }
};
