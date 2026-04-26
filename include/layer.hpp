#ifndef TINY_LAYER_HPP
#define TINY_LAYER_HPP

#include "tensor.hpp"
#include <memory>

namespace tiny_infer {

class Layer {
public:
    virtual ~Layer() = default;

    // 前向计算
    virtual Tensor forward(const Tensor& input) = 0;

    virtual std::string name() const = 0;
};

class LinearLayer : public Layer {
public:
    LinearLayer(int in_features, int out_features);

    Tensor forward(const Tensor& input) override;

    std::string name() const override { return "Linear"; }

    void set_weights(const Tensor& w) { weights_ = w; }
    void set_bias(const Tensor& b) { bias_ = b; }

    Tensor& weights() { return weights_; }
    Tensor& bias() { return bias_; }
private:
    Tensor weights_;
    Tensor bias_;
};

class ReLULayer : public Layer {
public:
    ReLULayer() = default;

    Tensor forward(const Tensor& input) override;

    std::string name() const override { return "ReLU"; }
};

class Sequential : public Layer {
public:
    Sequential() = default;

    void add(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
    }

    Tensor forward(const Tensor& input) override;

    std::string name() const override { return "Sequential"; }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};
} // namespace tiny_infer

#endif