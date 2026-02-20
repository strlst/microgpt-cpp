#include "model.hpp"
#include "value.hpp"
#include <cassert>
#include <iostream>

Model::Model(size_t vocab_size) {
    weights["wte"] = initialize_matrix(vocab_size, n_embed);
    weights["wpe"] = initialize_matrix(block_size, n_embed);
    weights["lm_head"] = initialize_matrix(vocab_size, n_embed);
    for (int i = 0; i < n_layer; ++i) {
        std::string prefix = "layer" + std::to_string(i) + "_";
        weights[prefix + "attn_wq"] = initialize_matrix(n_embed, n_embed);
        weights[prefix + "attn_wk"] = initialize_matrix(n_embed, n_embed);
        weights[prefix + "attn_wv"] = initialize_matrix(n_embed, n_embed);
        weights[prefix + "attn_wo"] = initialize_matrix(n_embed, n_embed);
        weights[prefix + "mlp_fc1"] = initialize_matrix(4 * n_embed, n_embed);
        weights[prefix + "mlp_fc2"] = initialize_matrix(n_embed, 4 * n_embed);
    }

    std::cout << "Initialized weights [";
    for (auto& [key, val] : weights)
        std::cout << key << " ";

    std::cout << "] with normdist(mean=" << dist_mean << ", std_dev=" << dist_std_dev << ")" << std::endl;
    std::cout << "Created model(n_embed=" << n_embed << ", n_head=" << n_head << ", n_layer=" << n_layer << ", head_dim=" << head_dim << ")" << std::endl;
}

void Model::infer(int BOS, size_t num_samples, float temperature) {
    std::cout << "Inferring " << num_samples << " samples with temperature " << temperature << std::endl;

    for (int step = 0; step < num_samples; step++) {
        std::vector<matrix_t> keys, values;
        prepare_tensors(keys, values, n_layer);

        int token_id = BOS;
        std::vector<int> sample;
        value_t temp_value = value_from(temperature);
        for (int pos_id = 0; pos_id < block_size; pos_id++) {
            // calculate logits like usual
            vector_t logits = gpt(token_id, pos_id, keys, values);

            // make our logits hotter (or colder, or even just warm)
            vector_t hot_logits;
            hot_logits.reserve(logits.size());
            for (auto& logit : logits)
                hot_logits.push_back(logit / temp_value);

            vector_t probabilities = softmax(hot_logits);

            std::vector<float> weights;
            weights.reserve(probabilities.size());
            float dist_sum = 0.f;
            for (auto& p : probabilities) {
                weights.push_back(p->data);
                dist_sum += p->data;
            }
            assert(dist_sum - 1.f < 1e-3);

            std::discrete_distribution<> discrete_dist(weights.begin(), weights.end());
            token_id = discrete_dist(generator);
            if (token_id == BOS)
                break;
            sample.push_back(token_id);
        }

        std::cout << "sample: ";
        for (auto s : sample)
            std::cout << (char)('a' + (char)s);
        std::cout << std::endl;
    }
}

matrix_t Model::initialize_matrix(int n_out, int n_in) {
    matrix_t matrix;
    matrix.reserve(n_out);

    // create properly sized vectors and initialize them randomly
    for (int y = 0; y < n_out; y++) {
        matrix.push_back(std::vector<value_t>());
        matrix.back().reserve(n_in);
        for (int x = 0; x < n_in; x++) {
            auto val = value_from((float)distribution(generator));
            matrix.back().push_back(val);
        }
    }

    return matrix;
}

vector_t Model::get_all_parameters() {
    vector_t parameters;
    // iterate all matrices and collect parameters
    for (auto& [key, matrix] : weights)
        for (auto& row : matrix)
            for (auto& val : row)
                parameters.push_back(val);
    return parameters;
}

vector_t Model::linear(const vector_t& x, const matrix_t& w) {
    vector_t ret;
    ret.reserve(w.size());
    for (const vector_t& row : w)
        ret.push_back(dot(row, x));
    return ret;
}

vector_t Model::softmax(vector_t& logits) {
    value_t max_value = max(logits);

    // construct exps vector and sum at the same time
    vector_t exponentials;
    exponentials.reserve(logits.size());
    value_t total = value_from(0.f);
    for (auto& logit : logits) {
        value_t exponential = (logit - max_value)->exp();
        exponentials.push_back(exponential);
        total = total + exponential;
    }

    // normalization step
    vector_t div;
    div.reserve(logits.size());
    for (auto& exponential : exponentials)
        div.push_back(exponential / total);

    return div;
}

vector_t Model::rms_norm(vector_t& x) {
    value_t length = value_from((float)x.size());
    value_t mean_square = dot(x, x) / length;
    value_t factor = value_from(1e-5f);
    value_t power = value_from(-.5f);
    value_t scale = (mean_square + factor)->pow(power);
    vector_t ret;
    ret.reserve(x.size());
    for (size_t i = 0; i < x.size(); i++)
        ret.push_back(x[i] * scale);
    return ret;
}

vector_t Model::gpt(int token_id, int pos_id, std::vector<matrix_t>& keys, std::vector<matrix_t>& values) {
    // load token embedding
    vector_t& token_emb = weights["wte"][token_id];
    // load position embedding
    vector_t& pos_emb = weights["wpe"][pos_id];
    assert(token_emb.size() == pos_emb.size());

    // compute embedding
    vector_t x;
    x.reserve(token_emb.size());
    for (size_t i = 0; i < token_emb.size(); i++)
        x.push_back(token_emb[i] + pos_emb[i]);

    // compute root-mean-square norm
    x = rms_norm(x);

    for (int li = 0; li < n_layer; li++) {
        //std::cout << "At layer " << li << std::endl;
        std::string prefix = "layer" + std::to_string(li) + "_";
        vector_t x_residual = x;
        x = rms_norm(x);
        vector_t q = linear(x, weights[prefix + "attn_wq"]);
        vector_t k = linear(x, weights[prefix + "attn_wk"]);
        vector_t v = linear(x, weights[prefix + "attn_wv"]);
        keys[li].push_back(k);
        values[li].push_back(v);

        const int seq_len = keys[li].size();
        const float inv_sqrt_d = 1.f / std::sqrt((float)head_dim);

        vector_t x_attn;
        x_attn.reserve(n_embed);

        for (int h = 0; h < n_head; h++) {
            const int hs = h * head_dim;

            // compute attention logits without copying slices:
            // score[t] = sum_j q[hs+j] * k_t[hs+j] (using dot_slice)
            vector_t attn_logits;
            attn_logits.reserve(seq_len);
            value_t inv_sqrt_val = value_from(inv_sqrt_d);
            for (int t = 0; t < seq_len; t++) {
                value_t score = dot_slice(q, hs, keys[li][t], hs, head_dim);
                attn_logits.push_back(score * inv_sqrt_val);
            }

            vector_t attn_weights = softmax(attn_logits);

            // weighted sum over value vectors (again using slice offsets)
            for (int j = 0; j < head_dim; j++) {
                value_t head_out = value_from(0.f);
                for (int t = 0; t < seq_len; t++)
                    head_out = head_out + (attn_weights[t] * values[li][t][hs + j]);
                x_attn.push_back(head_out);
            }
        }

        x = linear(x_attn, weights[prefix + "attn_wo"]);

        // residual add
        for (int i = 0; i < x.size(); i++)
            x[i] = x[i] + x_residual[i];
        x_residual = x;

        x = rms_norm(x);
        x = linear(x, weights[prefix + "mlp_fc1"]);

        // ReLU in-place (avoids a second vector allocation)
        for (auto& val : x)
            val = val->relu();

        x = linear(x, weights[prefix + "mlp_fc2"]);

        // residual add
        for (int i = 0; i < (int)x.size(); i++)
            x[i] = x[i] + x_residual[i];
    }

    vector_t logits = linear(x, weights["lm_head"]);
    return logits;
}
