#include <iostream>
#include "layer.hpp"
#include "tensor.hpp"

using namespace tiny_infer;

int main() {
    // 1. 定义网络
    auto fc1 = std::make_shared<LinearLayer>(784, 128);
    auto relu = std::make_shared<ReLULayer>();
    auto fc2 = std::make_shared<LinearLayer>(128, 10);

    // 2. 加载真实权重
    // 注意路径：如果你在 build 目录下运行，路径应该是 "../scripts/fc1_w.bin"
    try {
        fc1->weights().load_from_binary("../scripts/fc1_w.bin");
        fc1->bias().load_from_binary("../scripts/fc1_b.bin");
        fc2->weights().load_from_binary("../scripts/fc2_w.bin");
        fc2->bias().load_from_binary("../scripts/fc2_b.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    // 3. 模拟一个全 1 的输入（代表一张全白的 28x28 图片）
    Tensor input({1, 784});
    input.fill(1.0f);

    // 4. 推理
    Tensor hidden = fc1->forward(input);
    Tensor activated = relu->forward(hidden);
    Tensor output = fc2->forward(activated);

    std::cout << "Model Output (Class Probabilities):" << std::endl;
    output.display();

    return 0;
}