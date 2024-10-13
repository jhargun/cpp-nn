/*
ActivationFunction is an abstract class used to define an interface. All activation
functions should inherit from this class and implement the getActivation and
getActivationPrime functions. This forces all activation functions to
implement both their function and their derivative.
*/
#pragma once

template <typename T>
class ActivationFunction {
  protected:
    // Protected virtual functions used to follow NVI idiom
    virtual T getActivation(T x) const = 0;
    virtual T getActivationPrime(T x) const = 0;
  public:
    virtual ~ActivationFunction() = default;

    // These public functions are just wrappers around the protected virtual 
    // functions used to follow NVI idiom (for now)
    T activation(T x) const {
        return this->getActivation(x);
    }

    T activationPrime(T x) const {
        return this->getActivationPrime(x);
    }
};
