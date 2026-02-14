#ifndef __VALUE_HPP__
#define __VALUE_HPP__

#include <vector>
#include <cmath>

class Value {
private:
    // children of this node in the computation graph
    std::vector<Value*> children_;
    // local partial derivatives with respect to this node's childrens
    std::vector<float> local_grads_;
public:
    // store the actual value
    float data;
    // store the actual computed gradient
    float grad;

    Value(float data) {
        this->data = data;
        this->grad = 0;
    }

    // alternative constructor with pre-initialized children
    Value(float data, std::vector<Value*> children, std::vector<float> local_grads) {
        this->data = data;
        this->grad = 0;
        this->children_ = children;
        this->local_grads_ = local_grads;
    }

    Value operator+(float data) const {
        return Value(
            this->data + data,
            std::vector<Value*>{const_cast<Value*>(this),},
            std::vector<float>{1.f,}
        );
    }

    Value operator+(const Value& other) const {
        return Value(
            this->data + other.data,
            std::vector<Value*>{const_cast<Value*>(this), const_cast<Value*>(&other),},
            std::vector<float>{1.f, 1.f,}
        );
    }
    
    Value operator*(float data) const {
        return Value(
            this->data * data,
            std::vector<Value*>{const_cast<Value*>(this),},
            std::vector<float>{1.f,}
        );
    }

    Value operator*(const Value& other) const {
        return Value(
            this->data * other.data,
            std::vector<Value*>{const_cast<Value*>(this), const_cast<Value*>(&other),},
            std::vector<float>{other.data, this->data,}
        );
    }

    Value operator-() const {
        return (*this) * -1.f;
    }

    Value operator/(const Value& other) const {
        return (*this) * other.pow(-1);
    }

    Value operator-(float data) const {
        return (*this) + (-data);
    }

    Value operator-(const Value& other) const {
        return (*this) + (-other);
    }

    Value pow(float data) const {
        return Value(
            std::pow(this->data, data),
            std::vector<Value*>{const_cast<Value*>(this),},
            std::vector<float>{data * std::pow(this->data, data - 1),}
        );
    }

    Value log() const {
        return Value(
            std::log(this->data),
            std::vector<Value*>{const_cast<Value*>(this),},
            std::vector<float>{1.f / this->data,}
        );
    }

    Value exp() const {
        return Value(
            std::exp(this->data),
            std::vector<Value*>{const_cast<Value*>(this),},
            std::vector<float>{std::exp(this->data),}
        );
    }
};

#endif