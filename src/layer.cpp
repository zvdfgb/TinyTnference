#include "layer.hpp"

namespace tiny_infer {

LinearLayer::LinearLayer(int in_features, int out_features) 
    : weights_({in_features, out_features}), bias_({1, out_features}) {
    // 当前默认零初始化
}

Tensor LinearLayer::forward(const Tensor& input) {
    // 仿射变换：input * weights + bias
    Tensor temp = matmul(input, weights_);

    for (int i = 0; i < temp.shape()[0]; ++i) {
        for (int j = 0; j < temp.shape()[1]; ++j) {
            temp(i, j) += bias_(0, j);
        }
    }
    return temp;
}

Tensor ReLULayer::forward(const Tensor& input) {
    Tensor output(input.shape());

    int rows = input.shape()[0];
    int cols = input.shape()[1];
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = input(i, j);
            output(i, j) = std::max(0.0f, val);
        }
    }

    return output;
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor current_output = input;

    for (auto& layer : layers_) {
        current_output = layer->forward(current_output);
    }

    return current_output;
}
} // namespace tiny_infer