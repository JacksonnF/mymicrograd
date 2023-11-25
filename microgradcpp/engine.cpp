#include "engine.h"

#include <iostream>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>
#include <cmath>

Value::Value(float data, std::unordered_set<std::shared_ptr<Value>> _children) {
    this->data = data;
    this->grad = 0.0;
    this->_backward = []() -> void {};
    this->_children = std::move(_children);
};

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other){

    auto child_nodes = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>((this->data + other->data), child_nodes);

    out->_backward = [this, other, out]() -> void {
        this->grad += out->grad;
        other->grad += out->grad;
    };

    return out;
};
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator+(b);
}


std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {

    auto child_nodes = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(this->data * other->data, child_nodes);

    out->_backward = [this, other, out]() -> void {
        this->grad += out->grad * other->data;
        other->grad += out->grad * this->data;
    };

    return out;
};
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator*(b);
}


std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {

    auto child_nodes = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(this->data - other->data, child_nodes);

    out->_backward = [this, other, out]() -> void {
        this->grad -= out->grad;
        other->grad -= out->grad;
    };

    return out;
};
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator-(b);
}

std::shared_ptr<Value> Value::operator-() {
    return shared_from_this() * std::make_shared<Value>(-1.0);
};

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {

    auto child_nodes = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};
    auto out = std::make_shared<Value>(this->data / other->data, child_nodes);

    out->_backward = [this, other, out]() -> void {
        this->grad += out->grad / other->data;
        other->grad -= out->grad * this->data / (other->data * other->data);
    };

    return out;
};
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a->operator/(b);
}

std::shared_ptr<Value> Value::pow(float other) {

    auto child_nodes = std::unordered_set<std::shared_ptr<Value>>{shared_from_this()};
    auto out = std::make_shared<Value>(std::pow(this->data, other), child_nodes);

    out->_backward = [this, other, out]() -> void {
        this->grad += out->grad * std::pow(this->data, other - 1) * other;
    };

    return out;
};
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, float b) {
    return a->pow(b);
};

std::shared_ptr<Value> relu(const std::shared_ptr<Value>& a) {
    auto child_nodes = std::unordered_set<std::shared_ptr<Value>>{a};
    float result;
    if (a->data < 0.0) {
        result = 0.0;
    }
    else {
        result = a->data;
    }
    auto out = std::make_shared<Value>(result, child_nodes);

    out->_backward = [a, out]() -> void {
        if (a->data < 0.0) {
            a->grad += 0;
        }
        else {
            a->grad += out->grad;
        }
    };

    return out;
};

std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& a) {
    auto t = (std::exp(2 * a->data) - 1)/(std::exp(2 * a->data) + 1);
    auto child_nodes = std::unordered_set<std::shared_ptr<Value>>{a};

    auto out = std::make_shared<Value>(t, child_nodes);

    out->_backward = [a, out]() -> void {
        a->grad += out->grad * (1 - std::pow((std::exp(2 * a->data) - 1)/(std::exp(2 * a->data) + 1), 2.0));
    };

    return out;
};


void Value::backward() {

    std::vector<std::shared_ptr<Value>> nodes;
    std::unordered_set<std::shared_ptr<Value>> visited;

    topo_sort(shared_from_this(), visited, nodes);

    this->grad = 1.0;

    for (int i = nodes.size() - 1; i >= 0; i--) {
        nodes[i]->_backward();
    }

};

void Value::topo_sort(const std::shared_ptr<Value>& node, std::unordered_set<std::shared_ptr<Value>>& visited, std::vector<std::shared_ptr<Value>>& topo) {
    if (visited.find(node) == visited.end()) {

        visited.insert(node);
        for (auto child : node->_children) {

            topo_sort(child, visited, topo);
        }
        topo.push_back(node);
    }
};