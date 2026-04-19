#include "tensor.hpp"
#include <algorithm> // for std::fill
#include <iostream>
#include <fstream>
#include <omp.h>

namespace tiny_infer {

//两种构造函数：支持通过 {28, 28}和vector初始化
Tensor::Tensor(std::initializer_list<int> shape) : shape_(shape) 
{
    int total_size = 1;
    for (int s : shape_) total_size *= s;
    data_.resize(total_size, 0.0f); // 预分配内存并初始化为 0
}
Tensor::Tensor(const std::vector<int>& shape) : shape_(shape)
{
    int total_size = 1;
    for (int s : shape_) total_size *= s;
    data_.resize(total_size, 0.0f); // 预分配内存并初始化为 0
}

// 重载括号运算符实现 索引访问
float& Tensor::operator()(int r, int c) 
{
    // 假设我们目前只处理 2D 张量
    return data_[r * shape_[1] + c];
}
float Tensor::operator()(int r, int c) const
{
    return data_[r * shape_[1] + c];
}


// 辅助功能：填充数据
void Tensor::fill(float value)
{
    std::fill(data_.begin(), data_.end(), value);
}

//打印
void Tensor::display() const 
{
    for (int r = 0; r < shape_[0]; ++r) {
        for (int c = 0; c < shape_[1]; ++c) {
            std::cout << (*this)(r, c) << " ";
        }
        std::cout << std::endl;
    }
}



void Tensor::load_from_binary(const std::string& path) {
    // 1. 以二进制模式打开文件
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }

    // 2. 检查文件大小是否与我们的 Tensor 空间匹配
    // 计算我们 Tensor 需要多少字节：元素个数 * 4字节(float)
    size_t expected_bytes = data_.size() * sizeof(float);
    
    // 3. 直接将文件内容读取到 data_ 向量的内存首地址
    // data_.data() 返回底层数组的指针
    file.read(reinterpret_cast<char*>(data_.data()), expected_bytes);

    if (!file) {
        throw std::runtime_error("Error occurred while reading file: " + path);
    }

    file.close();
    std::cout << "Successfully loaded weights from: " << path << std::endl;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // 1. 维度检查
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Matrix dimensions mismatch for matmul!");
    }

    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];

    // 2. 创建结果张量
    Tensor result({M, N});
    
    // 务必确保 result 的初始值为 0，因为接下来的逻辑是 +=
    // （如果你的 Tensor 构造函数默认已经置 0，这一步可以省略，但写上更严谨）
    for (int i = 0; i < M * N; ++i) {
        // 假设你可以直接访问底层 vector，或者通过双重循环赋值
        result(i / N, i % N) = 0.0f; 
    }

    // 3. 工业级优化：i-k-j 顺序 + OpenMP 多核并行
    // 如果启用了 OpenMP，这行编译指令会让最外层的 i 循环被分配到你电脑的多个 CPU 核心上同时运行
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // 将 A 矩阵的标量提取出来，放在内层循环外面，减少重复访存
            float temp_a = a(i, k);
            
            // 最内层循环：连续访问 B 矩阵的第 k 行，并连续写入 Result 的第 i 行
            // 此时 CPU 的 Cache 预取机制 (Prefetching) 会达到最高效率
            for (int j = 0; j < N; ++j) {
                result(i, j) += temp_a * b(k, j);
            }
        }
    }

    return result;
}


void Tensor::draw_ascii() const {
    // 假设 Tensor 形状是 (1, 784)
    int width = 28;
    int height = 28;

    std::cout << "\n--- Visualizing Input Digit ---" << std::endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // 计算在一维数组中的偏移量
            float val = data_[i * width + j];
            
            // 根据像素强度打印不同字符
            if (val > 0.8f)      std::cout << "##"; // 黑色笔画
            else if (val > 0.2f) std::cout << ".."; // 灰色边缘
            else                 std::cout << "  "; // 白色背景
        }
        std::cout << std::endl; // 换行
    }
    std::cout << "-------------------------------\n" << std::endl;
}

// 标准化函数：对 Tensor 中的每个元素进行标准化处理
void Tensor::normalize(float mean, float stddev) {
    for (float& val : data_) {
        val = (val - mean) / stddev;
    }
}
//实现其他函数（后续更新。。）
}