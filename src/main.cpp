#include <iostream>
#include "tensor.hpp"

// 使用我们自定义的命名空间
using namespace tiny_infer;

int main() {
    // 1. 创建一个 3x3 的张量 (矩阵)
    // 这里的 {3, 3} 会触发我们写的 std::initializer_list 构造函数
    Tensor t({3, 3});

    std::cout << "Successfully created a 3x3 Tensor." << std::endl;
    std::cout << "Initial size: " << t.size() << " elements." << std::endl;

    // 2. 手动填充一些数据
    // 验证我们重载的 operator() 是否工作
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            t(i, j) = static_cast<float>(i * 3 + j + 1); // 填充 1, 2, 3... 9
        }
    }

    // 3. 调用 display 函数观察结果
    std::cout << "Tensor data content:" << std::endl;
    t.display();

    // 4. 简单的边界测试（可选）
    std::cout << "Value at (1, 1) is: " << t(1, 1) << " (Expected: 5)" << std::endl;

    return 0;
}