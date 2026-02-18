#include "adam.hpp"
#include "value.hpp"
#include <iostream>
#include <cmath>

Adam::Adam(int num_steps) {
    this->num_steps = num_steps;
}

void Adam::train(Model model, std::vector<std::string> docs, int BOS) {
    // commence training
    std::cout << "Training with num_steps=" << num_steps << std::endl;
    for (int step = 0; step < num_steps; step++) {
        std::string doc = docs[step % docs.size()];
        std::vector<int> tokens{BOS,};
        for (char ch : doc)
            tokens.push_back(ch - 'a');
        tokens.push_back(BOS);

        // create tensors
        std::vector<matrix_t> keys;
        std::vector<matrix_t> values;
        keys.reserve(model.n_layer);
        values.reserve(model.n_layer);
        for (int i = 0; i < model.n_layer; i++) {
            matrix_t empty, empty2;
            keys.push_back(empty);
            values.push_back(empty2);
        }

        vector_t losses;
        int n = std::min((size_t)model.block_size, tokens.size() - 1);
        for (int pos_id = 0; pos_id < n; pos_id++) {
            int token_id = tokens[pos_id];
            int target_id = tokens[pos_id + 1];
            vector_t logits = model.gpt(token_id, pos_id, keys, values);
            vector_t probs = model.softmax(logits);
            value_t loss_t = probs[target_id]->log()->operator-();
            losses.push_back(loss_t);
        }
        value_t loss = value_from((1.f / (float)n)) * sum(losses);

        // finally perform backwards pass
        loss->backward();
    }
}