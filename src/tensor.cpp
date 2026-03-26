#include "tensor.hpp"

namespace tiny_infer {

Tensor::Tensor(std::initializer_list<int> shape) : shape_(shape) {
    int total_size = 1;
    for (int s : shape_) total_size *= s;
    data_.resize(total_size, 0.0f); // 预分配内存并初始化为 0
}

float& Tensor::operator()(int r, int c) {
    // 假设我们目前只处理 2D 张量
    return data_[r * shape_[1] + c];
}

// ... 实现其他函数
}