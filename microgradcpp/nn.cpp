#include "nn.h"
#include "engine.h"

#include <vector>
#include <iostream>
#include <random>


Neuron::Neuron(int n_inputs) {
    // Initialize weights
    // number of weights = number of inputs as we perform a dot product
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> d(0.0f, 1.0f);

    for (int i = 0; i < n_inputs; i++) {
        this->weights.push_back(std::make_shared<Value>(d(gen)));
    }
    this->bias = std::make_shared<Value>(d(gen));
};

std::shared_ptr<Value> Neuron::forward(std::vector<std::shared_ptr<Value>>& inputs) {
    // Perform dot product
    auto out = std::make_shared<Value>(0.0);

    for (auto i = 0; i < inputs.size(); i++) {
        out = out + (inputs[i] * this->weights[i]);
    }

    return tanh(out + this->bias);
    // return out + this->bias;
};

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (auto weight : this->weights) {
        params.push_back(weight);
    }
    params.push_back(this->bias);

    return params;
};

Layer::Layer(int n_inputs, int n_neurons) {
    // Initialize neurons
    for (int i = 0; i < n_neurons; i++) {
        this->neurons.push_back(Neuron(n_inputs));
    }
};

std::vector<std::shared_ptr<Value>> Layer::forward(std::vector<std::shared_ptr<Value>>& inputs) {
    // Perform forward pass
    std::vector<std::shared_ptr<Value>> outs;

    for (auto neuron : this->neurons) {
        outs.push_back(neuron.forward(inputs));
    }

    return outs;
};

std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (auto neuron : this->neurons) {
        for (auto param : neuron.parameters()) {
            params.push_back(param);
        }
    }

    return params;
};

MLP::MLP(int input_size, std::vector<int> hidden_sizes, int output_size) {
    // Initialize layers
    this->layers.push_back(Layer(input_size, hidden_sizes[0]));

    for (int i = 1; i < hidden_sizes.size(); i++) {
        this->layers.push_back(Layer(hidden_sizes[i - 1], hidden_sizes[i]));
    }

    this->layers.push_back(Layer(hidden_sizes[hidden_sizes.size() - 1], output_size));
};

std::vector<std::shared_ptr<Value>> MLP::forward(std::vector<float>& inputs) {
    // Perform forward pass
    std::vector<std::shared_ptr<Value>> outs;

    // Convert inputs to Values
    std::vector<std::shared_ptr<Value>> vals;
    for (auto input : inputs) {
        vals.push_back(std::make_shared<Value>(input));
    }

    for (auto layer : this->layers) {
        outs = layer.forward(vals);
        vals = outs;
    }

    return outs;
};

std::vector<std::shared_ptr<Value>> MLP::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (auto layer : this->layers) {
        for (auto param : layer.parameters()) {
            params.push_back(param);
        }
    }

    return params;
};

