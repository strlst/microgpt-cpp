#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <random>
#include <map>
#include "value.hpp"

class Model {
private:
    // weight matrices stored inside dictionary
    std::map<std::string, matrix_t> weights;

    // normal dist params
    const unsigned dist_seed = 42;
    double dist_mean = 0.0;
    double dist_std_dev = 0.08;
    std::default_random_engine generator{};
    std::normal_distribution<double> distribution{dist_mean, dist_std_dev};
public:
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

    // constructor
    Model(size_t vocab_size);

    // model inference
    void infer(int BOS, size_t num_samples, float temperature = .5f);

    // model definition related functions
    matrix_t initialize_matrix(int n_out, int n_in);
    vector_t get_all_parameters();
    vector_t linear(const vector_t& x, const matrix_t& w);
    vector_t softmax(vector_t& logits);
    vector_t rms_norm(vector_t& x);
    vector_t gpt_old(int token_id, int pos_id, std::vector<matrix_t>& keys, std::vector<matrix_t>& values);
    vector_t gpt(int token_id, int pos_id, std::vector<matrix_t>& keys, std::vector<matrix_t>& values);
};

#endif