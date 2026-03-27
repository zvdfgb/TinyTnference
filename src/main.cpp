
#include "tensor.hpp"
#include <iostream>

using namespace tiny_infer;

int main() {
    Tensor a({2, 2});
    a(0, 0) = 1; a(0, 1) = 2;
    a(1, 0) = 3; a(1, 1) = 4;

    Tensor b({2, 2});
    b(0, 0) = 1; b(0, 1) = 0;
    b(1, 0) = 0; b(1, 1) = 1; // 单位矩阵

    try {
        Tensor c = matmul(a, b);
        std::cout << "Matmul Result:" << std::endl;
        c.display();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}