#ifndef __VALUE_HPP__
#define __VALUE_HPP__

#include <memory>
#include <unordered_set>
#include <vector>

class Value : public std::enable_shared_from_this<Value> {
    typedef std::shared_ptr<Value> value_t;
    typedef std::vector<value_t> vector_t;
    typedef std::unordered_set<value_t> set_t;
    typedef std::vector<vector_t> matrix_t;
private:
    // children of this node in the computation graph
    vector_t children;
    // local partial derivatives with respect to this node's childrens
    std::vector<float> local_grads;

    void build_topology(value_t node, vector_t& topology, std::unordered_set<Value*>& visited);
public:
    // store the actual value
    float data;
    // store the actual computed gradient
    float grad;

    Value(float data, vector_t children = {}, std::vector<float> local_grads = {});
    Value(float data, float grad, vector_t children = {}, std::vector<float> local_grads = {});

    // helper
    value_t copy();

    // binary ops
    value_t operator+(const value_t& other);
    value_t operator*(const value_t& other);
    value_t operator-(const value_t& other);
    value_t operator/(const value_t& other);
    value_t pow(const value_t& other);

    // unary ops
    value_t operator-();
    value_t exp();
    value_t log();
    value_t relu();

    void backward();

    // auxiliary overloads
    std::string to_string() const;
};

// export useful typedefs
typedef std::shared_ptr<Value> value_t;
typedef std::vector<value_t> vector_t;
typedef std::unordered_set<value_t> set_t;
typedef std::vector<vector_t> matrix_t;

// binary ops
value_t operator+(const value_t& lhs, const value_t& rhs);
value_t operator*(const value_t& lhs, const value_t& rhs);
value_t operator-(const value_t& lhs, const value_t& rhs);
value_t operator/(const value_t& lhs, const value_t& rhs);
value_t pow(const value_t& lhs, const value_t& rhs);

// helpers
value_t value_from(float x);
void print_vector_ptrs(const vector_t& vector);
void print_vector(const vector_t& vector);
void print_matrix(const matrix_t& matrix);
vector_t copy(vector_t& vec);
void prepare_tensors(std::vector<matrix_t>& keys, std::vector<matrix_t>& values, int n_layer);

// reduction ops
// NOTE: better defined as vector class ops?
value_t max(const vector_t& vec);
value_t sum(const vector_t& vec);
value_t dot(const vector_t& a, const vector_t& b);
value_t dot_slice(const vector_t& a_full, int a_offset, const vector_t& b_full, int b_offset, int len);

#endif