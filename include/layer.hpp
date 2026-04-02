#ifndef TINY_LAYER_HPP
#define TINY_LAYER_HPP

#include "tensor.hpp"
#include <memory>

namespace tiny_infer {

class Layer {
public:
    virtual ~Layer() = default;

    // 纯虚函数：要求所有子类必须实现自己的前向计算逻辑
    // 输入一个 Tensor，输出一个 Tensor
    virtual Tensor forward(const Tensor& input) = 0;

    // 获取层的名字（方便调试）
    virtual std::string name() const = 0;
};

class LinearLayer : public Layer {
public:
    // 构造函数：初始化权重和偏置的维度
    LinearLayer(int in_features, int out_features);

    // 实现前向传播：Output = Input * Weights + Bias
    Tensor forward(const Tensor& input) override;

    std::string name() const override { return "Linear"; }

    // 提供一个方法来手动设置权重（后面我们要从模型文件读取）
    void set_weights(const Tensor& w) { weights_ = w; }
    void set_bias(const Tensor& b) { bias_ = b; }

private:
    Tensor weights_; // 维度: (in_features, out_features)
    Tensor bias_;    // 维度: (1, out_features)
};

class ReLULayer : public Layer {
public:
    // ReLU 不需要参数，使用默认构造函数即可
    ReLULayer() = default;

    // 实现前向传播：Output = max(0, Input)
    Tensor forward(const Tensor& input) override;

    std::string name() const override { return "ReLU"; }
};

// 这个类是我们后续构建网络的容器，可以把多个层串联起来
class Sequential : public Layer {
public:
    Sequential() = default;

    // 添加层到容器中
    // 使用 std::shared_ptr 方便管理内存，防止内存泄漏
    void add(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
    }

    // 实现前向传播：一键跑通所有层
    Tensor forward(const Tensor& input) override;

    std::string name() const override { return "Sequential"; }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};
} // namespace tiny_infer

#endif