#include "tensor.hpp"
#include <algorithm> // for std::fill
#include <iostream>


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


Tensor matmul(const Tensor& a, const Tensor& b)
{

    // 1. 维度检查
    if (a.shape()[1] != b.shape()[0]) {
        throw std::runtime_error("无法计算矩阵乘法：A的列数必须等于B的行数");
    }
    
    int m = a.shape()[0];
    int n = a.shape()[1];
    int p = b.shape()[1];

    Tensor result({m, p});
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}
//实现其他函数（后续更新。。）
}