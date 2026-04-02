#include <iostream>
#include "layer.hpp"

using namespace tiny_infer;

int main() {
    // 1. 创建 Sequential 容器
    auto model = std::make_shared<Sequential>();

    // 2. 组装网络：Linear -> ReLU -> Linear
    // 第一层：2输入 -> 4输出
    model->add(std::make_shared<LinearLayer>(2, 4));
    // 第二层：激活
    model->add(std::make_shared<ReLULayer>());
    // 第三层：4输入 -> 1输出（回归结果）
    model->add(std::make_shared<LinearLayer>(4, 1));

    // 3. 模拟输入 (1, 2)
    Tensor input({1, 2});
    input.fill(0.5f);

    // 4. 一键推理！
    std::cout << "Running full model inference..." << std::endl;
    Tensor result = model->forward(input);

    std::cout << "Final Output:" << std::endl;
    result.display();

    return 0;
}