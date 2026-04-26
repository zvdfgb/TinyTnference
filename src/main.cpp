#include <iostream>
#include <memory>
#include <vector>
#include "tensor.hpp"
#include "layer.hpp"
#include <chrono>

using namespace tiny_infer;

int main() {
    try {
        // MLP: 784 -> 256 -> 128 -> 10
        auto fc1 = std::make_shared<LinearLayer>(784, 256);
        auto fc2 = std::make_shared<LinearLayer>(256, 128);
        auto fc3 = std::make_shared<LinearLayer>(128, 10);
        auto relu = std::make_shared<ReLULayer>();

        auto model = std::make_shared<Sequential>();
        model->add(fc1);
        model->add(relu);
        model->add(fc2);
        model->add(relu);
        model->add(fc3);

        // 加载权重
        fc1->weights().load_from_binary("../scripts/fc1_w.bin");
        fc1->bias().load_from_binary("../scripts/fc1_b.bin");
        fc2->weights().load_from_binary("../scripts/fc2_w.bin");
        fc2->bias().load_from_binary("../scripts/fc2_b.bin");
        fc3->weights().load_from_binary("../scripts/fc3_w.bin");
        fc3->bias().load_from_binary("../scripts/fc3_b.bin");

        // 加载输入批次
        const int BATCH_SIZE = 10;
        Tensor input_batch({BATCH_SIZE, 784});
        input_batch.load_from_binary("../scripts/batch_test.bin");

        // 保留输入可视化
        input_batch.draw_ascii();

        // 与训练阶段一致的标准化参数
        input_batch.normalize(0.1307f, 0.3081f);

        auto start_time = std::chrono::high_resolution_clock::now();

        Tensor output = model->forward(input_batch);
        auto end_time = std::chrono::high_resolution_clock::now();

        // 推理耗时（ms）
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Inference Time: " << duration.count() << " ms" << std::endl;
        std::cout << "--------------------------------------" << std::endl;

        // 按行取最大值作为类别
        for (int b = 0; b < BATCH_SIZE; ++b) {
            int pred = -1;
            float max_s = -1e9;
            for (int i = 0; i < 10; ++i) {
                if (output(b, i) > max_s) {
                    max_s = output(b, i);
                    pred = i;
                }
            }
            std::cout << "第 " << b << " 张图预测结果: " << pred << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "\033[31m[ERROR]\033[0m " << e.what() << std::endl;
        return -1;
    }

    return 0;
}