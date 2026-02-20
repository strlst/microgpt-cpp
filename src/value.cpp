#include <cassert>
#include <iomanip>
#include <iostream>
#include <memory>
#include <cmath>
#include <ostream>
#include <sstream>
#include <string>

#include "value.hpp"

Value::Value(float data, vector_t children, std::vector<float> local_grads) {
    this->data = data;
    this->grad = 0.f;
    this->children = children;
    this->local_grads = local_grads;
}

Value::Value(float data, float grad, vector_t children, std::vector<float> local_grads) {
    this->data = data;
    this->grad = grad;
    this->children = children;
    this->local_grads = local_grads;
}

value_t Value::copy() {
    return std::make_shared<Value>(data, grad, children, local_grads);
}

// binary ops
value_t Value::operator+(const value_t& other) {
    auto out_children = vector_t{shared_from_this(), other};
    auto out_grads = std::vector<float>{1.f, 1.f};
    auto out = std::make_shared<Value>(data + other->data, out_children, out_grads);
    return out;
}

value_t Value::operator*(const value_t& other) {
    auto out_children = vector_t{shared_from_this(), other};
    auto out_grads = std::vector<float>{other->data, this->data};
    auto out = std::make_shared<Value>(data * other->data, out_children, out_grads);
    return out;
}

value_t Value::operator-(const value_t& other) {
    // make use of overloaded global operator (lhs - rhs)
    return shared_from_this() + (other->operator-());
}

value_t Value::operator/(const value_t& other) {
    // make use of overloaded global operator (lhs - rhs)
    return shared_from_this() * (other->pow(std::make_shared<Value>(-1.f)));
}

value_t Value::pow(const value_t& other) {
    auto out_children = vector_t{shared_from_this()};
    // high school math: dx**n/dx = n * x**(n - 1)
    auto out_grads = std::vector<float>{other->data * std::pow(this->data, other->data - 1.f)};
    auto out = std::make_shared<Value>(std::pow(data, other->data), out_children, out_grads);
    return out;
}

// unary ops
value_t Value::operator-() {
    return shared_from_this() * std::make_shared<Value>(-1.f);
}

value_t Value::exp() {
    auto out_children = vector_t{shared_from_this()};
    // high school math: dexp(x)/dx = exp(x)
    auto out_grads = std::vector<float>{std::exp(this->data)};
    auto out = std::make_shared<Value>(std::exp(data), out_children, out_grads);
    return out;
}

value_t Value::log() {
    auto out_children = vector_t{shared_from_this()};
    // high school math: dlog(x)/dx = 1 / x
    auto out_grads = std::vector<float>{1 / this->data};
    auto out = std::make_shared<Value>(std::log(data), out_children, out_grads);
    return out;
}

value_t Value::relu() {
    auto out_children = vector_t{shared_from_this()};
    // high school math: dlog(x)/dx = 1 / x
    auto out_grads = std::vector<float>{(float)(this->data > 0)};
    auto out = std::make_shared<Value>(std::max(this->data, 0.f), out_children, out_grads);
    return out;
}

void Value::build_topology(vector_t& topology, set_t& visited) {
    value_t node = shared_from_this();
    // skip visited nodes
    if (visited.find(node) != visited.end())
        return;
    visited.emplace(node);
    for (value_t child : node->children) {
        child->build_topology(topology, visited);
    }
    topology.push_back(node);
}

void Value::backward() {
    vector_t topology;
    set_t visited;

    // build topology first
    this->build_topology(topology, visited);

    // reset root grad
    this->grad = 1.f;

    // iterate topology backwards for root-first grad
    for (auto it = topology.rbegin(); it != topology.rend(); it++) {
        auto& children = it->get()->children;
        auto& local_grads = it->get()->local_grads;
        // sanity check
        assert(children.size() == local_grads.size());
        // accumulate gradients
        for (int i = 0; i < children.size(); i++)
            children[i]->grad += it->get()->grad * local_grads[i];
    }
}

std::string Value::to_string() const {
    std::ostringstream str;
    str << std::fixed << std::setprecision(2);
    str << data;
    str << "{d=" << grad << "}";
    return str.str();
}

// non-class members

std::ostream& operator<<(std::ostream& os, const value_t& value) {
    os << value->to_string();
    return os;
}

value_t operator+(const value_t& lhs, const value_t& rhs) {
    return lhs->operator+(rhs);
}

value_t operator*(const value_t& lhs, const value_t& rhs) {
    return lhs->operator*(rhs);
}

value_t operator-(const value_t& lhs, const value_t& rhs) {
    return lhs->operator-(rhs);
}

value_t operator/(const value_t& lhs, const value_t& rhs) {
    return lhs->operator/(rhs);
}

value_t value_from(float x) {
    return std::make_shared<Value>(x);
}

void print_vector_ptrs(const vector_t& vector) {
    std::cout << "vector_ptrs[";
    for (auto& val : vector)
        std::cout << &val << ' ';
    std::cout << "]" << std::endl;
}

void print_vector(const vector_t& vector) {
    std::cout << "vector[";
    for (auto& val : vector)
        std::cout << val << ' ';
    std::cout << "]" << std::endl;
}

void print_matrix(const matrix_t& matrix) {
    std::cout << "matrix[";
    for (auto& row : matrix) {
        print_vector(row);
    }
    std::cout << "]" << std::endl;
}

vector_t copy(vector_t& vec) {
    vector_t new_vec;
    for (auto& v : vec)
        new_vec.push_back(v->copy());
    return new_vec;
}

value_t max(vector_t vec) {
    // compute max float
    float max_float = 0.f;
    for (auto& v : vec)
        max_float = std::max(max_float, v->data);
    // create shared value
    return value_from(max_float);
}

value_t sum(vector_t vec) {
    // perform sum
    value_t sum = value_from(0.f);
    for (auto& v : vec)
        sum = sum + v;
    return sum;
}

value_t dot(vector_t a, vector_t b) {
    // sanity check
    assert(a.size() == b.size());
    int n = b.size();

    // perform dot product
    value_t sum = value_from(0.f);
    for (int k = 0; k < n; k++) {
        sum = sum + (a[k] * b[k]);
    }

    return sum;
}
