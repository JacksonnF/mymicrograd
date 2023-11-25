#ifndef VALUE_H
#define VALUE_H

#include <functional>
#include <unordered_set>
#include <vector>
#include <memory>

class Value : public std::enable_shared_from_this<Value> {

private:

    void topo_sort(const std::shared_ptr<Value>& node, std::unordered_set<std::shared_ptr<Value>>& visited, std::vector<std::shared_ptr<Value>>& topo);

public:
    float data;
    float grad;
    
    std::function<void()> _backward; // Function used to update gradients of this Values children
    std::unordered_set<std::shared_ptr<Value>> _children; // Pointers toward "Children" or Parameters used to create this Value 

    Value(float data,  std::unordered_set<std::shared_ptr<Value>> _children = {}); // constructor
    void backward(); // Backpropagate throught the whole tree starting with this Value as the root

    // operator overloading for convenience. These will update the _backward function of the resulting Value
    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator-();
    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> pow(float other);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> relu(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& a);



#endif // VALUE_H
