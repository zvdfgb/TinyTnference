#include "layer.hpp"

namespace tiny_infer {

LinearLayer::LinearLayer(int in_features, int out_features) 
    : weights_({in_features, out_features}), bias_({1, out_features}) {
    // 实际项目中这里应该进行随机初始化，目前我们默认构造函数已经清零
}

Tensor LinearLayer::forward(const Tensor& input) {
    // 1. 执行矩阵乘法: temp = input * weights_
    Tensor temp = matmul(input, weights_);

    // 2. 加上偏置: res = temp + bias_
    // TODO: 为了让代码更简洁，我们可以给 Tensor 重载一个 + 运算符
    // 暂时我们可以手动写一个循环来加偏置
    for (int i = 0; i < temp.shape()[0]; ++i) {
        for (int j = 0; j < temp.shape()[1]; ++j) {
            temp(i, j) += bias_(0, j);
        }
    }
    return temp;
}

} // namespace tiny_infer