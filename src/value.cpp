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
    this->children = std::move(children);
    this->local_grads = std::move(local_grads);
}

Value::Value(float data, float grad, vector_t children, std::vector<float> local_grads) {
    this->data = data;
    this->grad = grad;
    this->children = std::move(children);
    this->local_grads = std::move(local_grads);
}

value_t Value::copy() {
    return std::make_shared<Value>(data, grad, children, local_grads);
}

// binary ops
value_t Value::operator+(const value_t& other) {
    auto out_children = vector_t{shared_from_this(), other};
    auto out_grads = std::vector<float>{1.f, 1.f};
    return std::make_shared<Value>(data + other->data, std::move(out_children), std::move(out_grads));
}

value_t Value::operator*(const value_t& other) {
    auto out_children = vector_t{shared_from_this(), other};
    auto out_grads = std::vector<float>{other->data, this->data};
    return std::make_shared<Value>(data * other->data, std::move(out_children), std::move(out_grads));
}

value_t Value::operator-(const value_t& other) {
    // make use of overloaded global operator (lhs + rhs)
    return shared_from_this() + (other->operator-());
}

value_t Value::operator/(const value_t& other) {
    // make use of overloaded global operator (lhs * rhs)
    return shared_from_this() * (other->pow(std::make_shared<Value>(-1.f)));
}

value_t Value::pow(const value_t& other) {
    auto out_children = vector_t{shared_from_this()};
    // high school math: dx**n/dx = n * x**(n - 1)
    auto out_grads = std::vector<float>{other->data * std::pow(this->data, other->data - 1.f)};
    return std::make_shared<Value>(std::pow(data, other->data), std::move(out_children), std::move(out_grads));
}

// unary ops
value_t Value::operator-() {
    return shared_from_this() * std::make_shared<Value>(-1.f);
}

value_t Value::exp() {
    auto out_children = vector_t{shared_from_this()};
    // high school math: dexp(x)/dx = exp(x)
    auto out_grads = std::vector<float>{std::exp(this->data)};
    return std::make_shared<Value>(std::exp(data), std::move(out_children), std::move(out_grads));
}

value_t Value::log() {
    auto out_children = vector_t{shared_from_this()};
    // high school math: dlog(x)/dx = 1 / x
    auto out_grads = std::vector<float>{1.f / this->data};
    return std::make_shared<Value>(std::log(data), std::move(out_children), std::move(out_grads));
}

value_t Value::relu() {
    auto out_children = vector_t{shared_from_this()};
    auto out_grads = std::vector<float>{(float)(this->data > 0)};
    return std::make_shared<Value>(std::max(this->data, 0.f), std::move(out_children), std::move(out_grads));
}

// use raw pointer to avoid shared_ptr ref-counts
void Value::build_topology(value_t node, vector_t& topology, std::unordered_set<Value*>& visited) {
    Value* raw = node.get();
    if (!visited.insert(raw).second)
        return;
    for (const value_t& child : node->children) {
        child->build_topology(child, topology, visited);
    }
    topology.push_back(std::move(node));
}

void Value::backward() {
    vector_t topology;
    std::unordered_set<Value*> visited;

    // build topology first
    build_topology(shared_from_this(), topology, visited);

    // reset root grad
    grad = 1.f;

    // iterate topology backwards for root-first grad
    for (auto it = topology.rbegin(); it != topology.rend(); it++) {
        Value* node = it->get();
        const float node_grad = node->grad;
        const auto& children = node->children;
        const auto& local_grads = node->local_grads;
        assert(children.size() == local_grads.size());
        for (size_t i = 0; i < children.size(); i++)
            children[i]->grad += node_grad * local_grads[i];
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
    new_vec.reserve(vec.size());
    for (auto& v : vec)
        new_vec.push_back(v->copy());
    return new_vec;
}

void prepare_tensors(std::vector<matrix_t>& keys, std::vector<matrix_t>& values, int n_layer) {
    keys.reserve(n_layer);
    values.reserve(n_layer);
    for (int i = 0; i < n_layer; i++) {
        keys.emplace_back();
        values.emplace_back();
    }
}

value_t max(const vector_t& vec) {
    float max_float = vec[0]->data;
    for (size_t i = 1; i < vec.size(); i++)
        max_float = std::max(max_float, vec[i]->data);
    return value_from(max_float);
}

value_t sum(const vector_t& vec) {
    value_t total = value_from(0.f);
    for (auto& v : vec)
        total = total + v;
    return total;
}

// optimized dot: fuses multiply-add in float, producing ONE value node only later.
// the node is initialized with all children.
// with 2n children and local grads, a lot of shared_ptr allocations are avoided and
// topology traversal is also made more efficient.
value_t dot(const vector_t& a, const vector_t& b) {
    assert(a.size() == b.size());
    const size_t n = a.size();

    // initializations
    float result = 0.f;
    vector_t children;
    std::vector<float> local_grads;
    children.reserve(2 * n);
    local_grads.reserve(2 * n);

    for (size_t k = 0; k < n; k++) {
        result += a[k]->data * b[k]->data;
        // grad of output w.r.t. a[k] is b[k]->data, and vice versa
        children.push_back(a[k]);
        children.push_back(b[k]);
        local_grads.push_back(b[k]->data);
        local_grads.push_back(a[k]->data);
    }

    return std::make_shared<Value>(result, std::move(children), std::move(local_grads));
}

// optimized dot for linears:
// operates on a contiguous slice [offset, offset+len) of x without
// materializing a new sub-vector
value_t dot_slice(const vector_t& a_full, int a_offset, const vector_t& b_full, int b_offset, int len) {
    assert(a_full.size() == b_full.size());
    const size_t n = a_full.size();

    // initializations
    float result = 0.f;
    vector_t children;
    std::vector<float> local_grads;
    children.reserve(2 * len);
    local_grads.reserve(2 * len);

    for (int j = 0; j < len; j++) {
        const value_t& a = a_full[a_offset + j];
        const value_t& b = b_full[b_offset + j];
        result += a->data * b->data;
        children.push_back(a);
        children.push_back(b);
        local_grads.push_back(b->data);
        local_grads.push_back(a->data);
    }
    return std::make_shared<Value>(result, std::move(children), std::move(local_grads));
}
