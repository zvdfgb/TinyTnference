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

Tensor ReLULayer::forward(const Tensor& input) {
    // 1. 创建一个与输入形状完全相同的 Tensor 来存储结果
    // 这里我们使用了 Tensor(const std::vector<int>& shape) 构造函数
    Tensor output(input.shape());

    // 2. 核心逻辑：逐元素应用 ReLU 函数
    // 虽然我们在 Tensor 外部用 (row, col) 访问，但在这里我们可以
    // 直接操作底层的 data_ vector，效率最高。
    // 这里我们需要给 Tensor 类增加一个 public 接口来获取 data_ 的大小和访问权限。

    // 暂时我们可以用我们在 Tensor 里写的 operator() 遍历：
    int rows = input.shape()[0];
    int cols = input.shape()[1];
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = input(i, j);
            // f(x) = max(0, x)
            output(i, j) = std::max(0.0f, val);
        }
    }

    return output;
}
} // namespace tiny_infer