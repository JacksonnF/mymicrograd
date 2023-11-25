#ifndef NN_H 
#define NN_H

#include <vector>
#include <memory>
#include "engine.h"

// class Module {
//     public:
//         void zero_grad();
//         virtual std::vector<std::shared_ptr<Value>> parameters();
// };

class Neuron {
    public:
        std::vector<std::shared_ptr<Value>> weights;
        std::shared_ptr<Value> bias;
        Neuron(int n_inputs);
        std::shared_ptr<Value> forward(std::vector<std::shared_ptr<Value>>& inputs);

        std::vector<std::shared_ptr<Value>> parameters();
};

class Layer {
    public:
        std::vector<Neuron> neurons;
        Layer(int n_inputs, int n_neurons);
        std::vector<std::shared_ptr<Value>> forward(std::vector<std::shared_ptr<Value>>& inputs);

        std::vector<std::shared_ptr<Value>> parameters();
};

class MLP {
    public:
        std::vector<Layer> layers;
        MLP(int input_size, std::vector<int> hidden_sizes, int output_size);
        std::vector<std::shared_ptr<Value>> forward(std::vector<float>& inputs);

        std::vector<std::shared_ptr<Value>> parameters();
};

#endif // NN_H
