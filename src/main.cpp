#include <iostream>
#include <memory>
#include <vector>
#include "tensor.hpp"
#include "layer.hpp"
#include <chrono> // 用于计时

using namespace tiny_infer;

int main() {
    try {
        // 1. 构建更加专业的多层感知机 (MLP) 结构
        // 结构：Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10)
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

        // 2. 加载专业版模型权重
        // 确保路径正确，指向你刚才运行 Python 脚本生成的 .bin 文件
        std::cout << "Loading weights..." << std::endl;
        fc1->weights().load_from_binary("../scripts/fc1_w.bin");
        fc1->bias().load_from_binary("../scripts/fc1_b.bin");
        fc2->weights().load_from_binary("../scripts/fc2_w.bin");
        fc2->bias().load_from_binary("../scripts/fc2_b.bin");
        fc3->weights().load_from_binary("../scripts/fc3_w.bin");
        fc3->bias().load_from_binary("../scripts/fc3_b.bin");

        // 3. 准备并加载真实的测试图片
        const int BATCH_SIZE = 10;
        Tensor input_batch({BATCH_SIZE, 784});
        input_batch.load_from_binary("../scripts/batch_test.bin");

        // [重要优化] 可视化输入，确认图片是否正确
        input_batch.draw_ascii();

        // [重要优化] 数据标准化处理
        // 这一步必须与 Python 训练时的参数 (mean=0.1307, std=0.3081) 完全一致
        std::cout << "Normalizing input data..." << std::endl;
        input_batch.normalize(0.1307f, 0.3081f);

       
        std::cout << "Inference running..." << std::endl;

        // 1. 记录开始时间
        auto start_time = std::chrono::high_resolution_clock::now();

        // 4. 执行前向推理
        Tensor output = model->forward(input_batch);
        // 3. 记录结束时间
        auto end_time = std::chrono::high_resolution_clock::now();

        // 4. 计算并打印耗时 (单位：毫秒)
        std::chrono::duration<double, std::milli> duration = end_time - start_time;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Inference Time: " << duration.count() << " ms" << std::endl;
        std::cout << "--------------------------------------" << std::endl;

        // 5. 解析结果：循环打印每一行的最大值
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
        // 异常处理：捕获文件不存在或维度不匹配等错误
        std::cerr << "\033[31m[ERROR]\033[0m " << e.what() << std::endl;
        return -1;
    }

    return 0;
}