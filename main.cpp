#include <iostream>
#include <vector>
#include "microgradcpp/engine.h"
#include "microgradcpp/nn.h"


int main()  {
    std::cout << "Hello World!" << std::endl;

    std::vector<int> v = {4, 4};

    auto n = MLP(3, {4, 4}, 1);

    std::vector<std::vector<float>> inputs = 
    {{2.0, 3.0, -1.0}, 
    {3.0, -1.0, 0.5},
    {0.5, 1.0, 1.0},
    {1.0, 1.0, -1.0}};

    std::vector<float> outputs = {1.0, -1.0, -1.0, 1.0};

    int epochs = 1000;
    float lr = 0.1;

    for (int k = 0; k < epochs; k++) {
        std::vector<std::shared_ptr<Value>> preds;
        for (auto input : inputs) {
            preds.push_back(n.forward(input)[0]);
        }

        auto loss = std::make_shared<Value>(0.0);
        for (int i = 0; i < preds.size(); i++) {
            std::cout << "Prediction, Real " << outputs[i] << " " << preds[i]->data << std::endl;
            loss = loss + pow(std::make_shared<Value>(outputs[i]) - preds[i], 2.0);
        }

        auto parameters_zero = n.parameters();
        for (auto pp : parameters_zero) {
            pp->grad = 0.0;
        }

        loss->backward();

        auto parameters = n.parameters();
        for (auto p : parameters) {
            // std::cout << "GRADIENTS " << p->grad << std::endl;
            p->data += -lr * p->grad;
        } 

        std::cout << "Loss: " << loss->data << std::endl;

    }
    std::cout << "Prediction (1.0 expected)" << n.forward(inputs[0])[0]->data << std::endl;

    return 0;
}
