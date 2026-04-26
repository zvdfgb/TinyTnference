#include "tensor.hpp"
#include <algorithm> // for std::fill
#include <iostream>
#include <fstream>
#include <omp.h>

namespace tiny_infer {

// 支持列表或向量两种形状初始化方式
Tensor::Tensor(std::initializer_list<int> shape) : shape_(shape) 
{
    int total_size = 1;
    for (int s : shape_) total_size *= s;
    data_.resize(total_size, 0.0f);
}
Tensor::Tensor(const std::vector<int>& shape) : shape_(shape)
{
    int total_size = 1;
    for (int s : shape_) total_size *= s;
    data_.resize(total_size, 0.0f);
}

// 二维索引访问
float& Tensor::operator()(int r, int c) 
{
    return data_[r * shape_[1] + c];
}
float Tensor::operator()(int r, int c) const
{
    return data_[r * shape_[1] + c];
}


// 填充全部元素
void Tensor::fill(float value)
{
    std::fill(data_.begin(), data_.end(), value);
}

// 输出二维张量
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
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }

    size_t expected_bytes = data_.size() * sizeof(float);

    file.read(reinterpret_cast<char*>(data_.data()), expected_bytes);

    if (!file) {
        throw std::runtime_error("Error occurred while reading file: " + path);
    }

    file.close();
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    // 矩阵乘法维度检查
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("Matrix dimensions mismatch for matmul!");
    }

    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];

    Tensor result({M, N});

    for (int i = 0; i < M * N; ++i) {
        result(i / N, i % N) = 0.0f; 
    }

    // i-k-j 顺序 + OpenMP 并行
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float temp_a = a(i, k);

            for (int j = 0; j < N; ++j) {
                result(i, j) += temp_a * b(k, j);
            }
        }
    }

    return result;
}


void Tensor::draw_ascii() const {
    const int width = 28;
    const int height = 28;
    const int pixels_per_image = width * height;

    if (shape_.size() != 2 || shape_[1] != pixels_per_image) {
        throw std::runtime_error("draw_ascii expects shape [batch, 784]");
    }

    const int batch = shape_[0];
    const int show_count = std::min(batch, 10);

    std::cout << "\n--- Visualizing Input Digits (" << show_count << "/" << batch << ") ---" << std::endl;
    for (int n = 0; n < show_count; ++n) {
        std::cout << "[Image " << n << "]" << std::endl;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float val = data_[n * pixels_per_image + i * width + j];

                if (val > 0.8f)      std::cout << "##";
                else if (val > 0.2f) std::cout << "..";
                else                 std::cout << "  ";
            }
            std::cout << std::endl;
        }
        std::cout << "-------------------------------" << std::endl;
    }
    std::cout << std::endl;
}

// 标准化：x = (x - mean) / stddev
void Tensor::normalize(float mean, float stddev) {
    for (float& val : data_) {
        val = (val - mean) / stddev;
    }
}
}