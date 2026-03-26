#ifndef TINY_TENSOR_HPP
#define TINY_TENSOR_HPP

#include <vector>
#include <initializer_list>
#include <iostream>

namespace tiny_infer {

class Tensor {
public:
    //构造函数：支持通过 {28, 28} 这种方式初始化
    Tensor(std::initializer_list<int> shape);
    Tensor(const std::vector<int>& shape);

    // 2基础信息获取
    const std::vector<int>& shape() const { return shape_; }
    int size() const { return data_.size(); }

    // 3索引访问（重载括号运算符）
    //  my_tensor(1, 2) 来访问数据
    float& operator()(int r, int c);
    float operator()(int r, int c) const;

    // 辅助功能：填充数据
    void fill(float value)
    {
        std::fill(data_.begin(), data_.end(), value);
    }
    void display() const // 打印出来看看结果
    {
        int rows = shape_[0];
        int cols = shape_[1];
        for (int i = 0; i < rows; ++i) 
        {
            for (int j = 0; j < cols; ++j) 
            {
                std::cout << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    std::vector<int> shape_;    // 存储维度，例如 {rows, cols}
    std::vector<float> data_;   // 实际存储数据的“大平层”内存
};

} // namespace tiny_infer

#endif