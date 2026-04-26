#ifndef TINY_TENSOR_HPP
#define TINY_TENSOR_HPP

#include <vector>
#include <initializer_list>
#include <iostream>

namespace tiny_infer {

class Tensor {
public:
    // 支持列表或向量初始化
    Tensor(std::initializer_list<int> shape);
    Tensor(const std::vector<int>& shape);

    const std::vector<int>& shape() const { return shape_; }
    int size() const { return data_.size(); }

    // 二维索引访问
    float& operator()(int r, int c);
    float operator()(int r, int c) const;

    void fill(float value);

    void display() const;

    void load_from_binary(const std::string& path);

    // ASCII 可视化
    void draw_ascii() const;

    // 标准化
    void normalize(float mean, float stddev);

private:
    std::vector<int> shape_;
    std::vector<float> data_;
};

Tensor matmul(const Tensor& a, const Tensor& b);

    
} // namespace tiny_infer

#endif