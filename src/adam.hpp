#ifndef __ADAM_HPP__
#define __ADAM_HPP__

#include <string>
#include <vector>

#include "model.hpp"

class Adam {
private:
    double learning_rate = 0.01;
    double beta1 = 0.85;
    double beta2 = 0.99;
    double eps_adam = 1e-8;
    int num_steps;
public:
    Adam(int num_steps = 1000);
    void train(Model& model, std::vector<std::string> docs, int BOS);
};

#endif