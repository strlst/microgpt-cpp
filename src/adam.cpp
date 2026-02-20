#include "adam.hpp"
#include "value.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

Adam::Adam(int num_steps) {
    this->num_steps = num_steps;
}

void Adam::train(Model model, std::vector<std::string> docs, int BOS) {
    // get params as a vector
    vector_t parameters = model.get_all_parameters();

    // initialize moment buffers (first and second moment)
    std::vector<double> mom(parameters.size(), 1.0);
    std::vector<double> vel(parameters.size(), 1.0);

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

        // Adam optimizer update: update the model parameters based on gradients
        double lr_t = learning_rate * (1. - ((float)step) / ((float)num_steps));
        for (int i = 0; i < parameters.size(); i++) {
            mom[i] = beta1 * mom[i] + (1. - beta1) * parameters[i]->grad;
            vel[i] = beta2 * vel[i] + (1. - beta2) * std::pow(parameters[i]->grad, 2);
            double m_hat = mom[i] / (1. - std::pow(beta1, step + 1));
            double v_hat = vel[i] / (1. - std::pow(beta2, step + 1));
            double update = lr_t * m_hat / (std::pow(v_hat, .5) - eps_adam);
            //std::cout << "updating param" << i << " by grad " << update << std::endl;
            parameters[i]->data -= (float)update;
            parameters[i]->grad = 0;
        }

        std::cout << "step " << std::setw(4) << step << " / " << num_steps << " | Loss " << loss->data << std::endl;
    }
}