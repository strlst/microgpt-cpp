#include <iostream>
#include "util.h"
#include "model.hpp"
#include "adam.hpp"

int main() {
    // open local file (or remote location if not downloaded)
    auto ifstream = open_url_cached("https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt");

    // fetch docs
    std::vector<std::string> docs;
    std::string line;
    while (std::getline(ifstream, line))
        docs.push_back(line);

    // let there be a tokenizer to translate string to discrete symbols and back
    std::set<char> unique_chars;
    for (std::string doc : docs) {
        for (char letter : doc) {
            unique_chars.insert(letter);
        }
    }
    int BOS = unique_chars.size();
    int vocab_size = unique_chars.size() + 1;
    std::cout << "Initialized vocabulary of size " << vocab_size << std::endl;

    // initialize model, especially the params, so there be stored values
    Model model(vocab_size);
    Adam adam(1);
    adam.train(model, docs, BOS);

    return 0;
}