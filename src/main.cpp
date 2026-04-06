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
        std::cerr << "Weight load error: " << e.what() << std::endl;
        return -1;
    }

    // 3. 准备输入数据
    Tensor input({1, 784});
    try {
        // 加载刚才 Python 导出的那个数字 7（或随机数字）
        input.load_from_binary("../scripts/test_digit.bin");
        std::cout << "Image loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Image load error: " << e.what() << std::endl;
        return -1;
    }


    // 4. 推理
    Tensor hidden = fc1->forward(input);
    Tensor activated = relu->forward(hidden);
    Tensor output = fc2->forward(activated);

    float max_score = -1e9;
    int predicted_label = -1;

    for (int i = 0; i < 10; ++i) {
        float score = output(0, i);
        std::cout << "Digit " << i << " score: " << score << std::endl;
        if (score > max_score) {
            max_score = score;
            predicted_label = i;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Final Prediction: This is digit " << predicted_label << "!" << std::endl;
    std::cout << "========================================" << std::endl;
    input.draw_ascii(); // 打印输出分数看看结果
    return 0;
}