#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <cassert>
#include <random>
#include <vector>
#include <map>
#include "value.hpp"

typedef std::vector<Value> vector_t;
typedef std::vector<vector_t> matrix_t;

class Model {
private:
    // embedding dimension
    int n_embed = 16;
    // number of attention heads
    int n_head = 4;
    // number of layers
    int n_layer = 1;
    // maximum sequence length
    int block_size = 16;
    // dimension of each head
    int head_dim = n_embed / n_head;

    // weight matrices stored inside dictionary
    std::map<std::string, matrix_t> weights;

    // normal dist params
    const unsigned dist_seed = 42;
    double dist_mean = 0.0;
    double dist_std_dev = 0.08;
    std::default_random_engine generator{};
    std::normal_distribution<double> distribution{dist_mean, dist_std_dev};
public:
    Model() {
        weights["attn_wq"] = initialize_matrix(n_embed, n_embed);
        weights["attn_wk"] = initialize_matrix(n_embed, n_embed);
        weights["attn_wv"] = initialize_matrix(n_embed, n_embed);
        weights["attn_wo"] = initialize_matrix(n_embed, n_embed);
        weights["mlp_fc1"] = initialize_matrix(4 * n_embed, n_embed);
        weights["mlp_fc2"] = initialize_matrix(n_embed, 4 * n_embed);
    }

    matrix_t initialize_matrix(int n_out, int n_in) {
        matrix_t matrix;
        matrix.reserve(n_out);

        for (int y = 0; y < n_out; y++) {
            matrix.push_back(std::vector<Value>());
            matrix.back().reserve(n_in);
            for (int x = 0; x < n_in; x++) {
                matrix.back().push_back(distribution(generator));
            }
        }

        return matrix;
    }

    std::vector<Value*> get_all_parameters() {
        std::vector<Value*> parameters;

        // iterate all matrices and collect parameters
        for (auto& [key, matrix] : weights) {
            for (auto& row : matrix) {
                for (auto& val : row) {
                    parameters.push_back(&val);
                }
            }
        }

        return parameters;
    }

    // TODO: omp parallelization?
    vector_t linear(const vector_t& x, const matrix_t& w) {
        vector_t ret;
        ret.reserve(w.size());

        // iterate weight rows
        for (const vector_t& row : w) {
            // sanity check
            assert(row.size() == x.size());

            // perform dot product
            Value sum(0);
            int n = x.size();
            for (int k = 0; k < n; k++) {
                sum = sum + (row[k] * x[k]);
            }
            ret.push_back(sum);
        }

        return ret;
    }

    vector_t softmax(vector_t logits) {
        // TODO: implement
        return vector_t();
    }

    vector_t rms_norm(vector_t x) {
        // TODO: implement
        return vector_t();
    }

    // TODO: vector_t gpt(...) {}
};

#endif