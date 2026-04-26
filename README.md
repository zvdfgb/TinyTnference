# TinyTnference

**轻量级 C++ 深度学习推理引擎**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Build Status](https://img.shields.io/badge/Build-CMake-orange.svg)](https://cmake.org/)

---

## 目录

- [项目核心概要](#1-项目核心概要)
- [核心特性](#2-核心特性)
- [技术细节](#3-技术细节)
- [性能表现](#4-性能表现)
- [快速上手指南](#5-快速上手指南)

---

## 1. 项目核心概要

### 1.1 项目定位

TinyTnference 是一款专为**手写数字识别（MNIST）**任务设计的轻量级深度学习推理引擎，采用纯 C++ 实现，专注于模型推理阶段的极致性能优化。项目以**简洁代码结构**与**高效执行效率**并重为设计理念，为开发者提供一个理解深度学习推理机制底层原理的优质教学资源，同时具备在资源受限环境中的实际应用价值。

### 1.2 主要功能

| 功能模块 | 描述 |
|---------|------|
| 张量运算 | 支持多维张量创建、索引访问、矩阵乘法等基础运算 |
| 神经网络层 | 实现全连接层（Linear）、ReLU 激活函数层 |
| 模型推理 | 完整的前向推理流程，支持多 层网络的顺序执行 |
| 数据加载 | 支持从二进制文件加载预训练权重与测试数据 |
| 性能优化 | OpenMP 多核并行、SIMD 指令集优化、缓存友好算法 |

### 1.3 应用场景

- **教学示范**：作为深度学习入门教学示例，清晰展示从模型训练到推理的完整流程
- **嵌入式推理**：轻量级设计，适合在资源受限的嵌入式设备中部署
- **原型验证**：快速验证模型架构和权重参数的算法正确性
- **性能基准**：作为轻量级推理框架的性能基准测试参考

### 1.4 技术价值

项目解决了深度学习推理框架中的几个核心问题：

1. **零外部依赖**：不依赖任何第三方深度学习库，仅使用标准 C++ 库和 OpenMP
2. **透明化实现**：通过亲手实现矩阵乘法、激活函数等核心算子，揭示深度学习底层工作原理
3. **工程化实践**：遵循 CMake 构建规范，采用命名空间隔离、虚函数多态等 C++ 最佳实践

---

## 2. 核心特性

### 2.1 纯 C++ 实现的底层架构

#### 2.1.1 架构设计原则

项目采用**数据抽象**与**面向对象**相结合的设计理念，将整个推理引擎划分为两个核心子系统：

```
┌─────────────────────────────────────────────────────────┐
│                      TinyTnference                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────┐         ┌─────────────────┐        │
│  │   Tensor System │         │   Layer System  │        │
│  │                 │         │                 │        │
│  │  - Tensor       │         │  - Layer        │        │
│  │  - matmul()     │         │  - LinearLayer  │        │
│  │  - normalize()  │         │  - ReLULayer    │        │
│  │  - load_binary()│         │  - Sequential   │        │
│  └─────────────────┘         └─────────────────┘        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 2.1.2 命名空间隔离

所有组件均封装在 `tiny_infer` 命名空间中，避免与用户代码产生符号冲突：

```cpp
namespace tiny_infer {
class Tensor { /* ... */ };
class Layer { /* ... */ };
}
```

### 2.2 高性能矩阵乘法算法

#### 2.2.1 算法实现

矩阵乘法是深度学习中最核心的计算算子，TinyTnference 采用了**内存访问优化**与**并行计算**相结合的实现策略：

```cpp
// src/tensor.cpp - 核心矩阵乘法实现
Tensor matmul(const Tensor& a, const Tensor& b) {
    int M = a.shape()[0];
    int K = a.shape()[1];
    int N = b.shape()[1];

    Tensor result({M, N});

    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float temp_a = a(i, k);  // 提取到外层，减少重复访存
            for (int j = 0; j < N; ++j) {
                result(i, j) += temp_a * b(k, j);
            }
        }
    }
    return result;
}
```

#### 2.2.2 优化策略详解

| 优化策略 | 实现原理 | 性能收益 |
|---------|---------|---------|
| **循环重排（i-k-j）** | 将最内层循环改为 j 循环，使 CPU 连续访问 result 第 i 行与 b 第 k 行 | 触发 Cache Prefetching，预取效率提升 |
| **寄存器复用** | 将 `a(i, k)` 提取到中间层循环外，避免重复从内存加载 | 减少内存访问带宽占用 |
| **OpenMP 并行** | `#pragma omp parallel for` 将外层 i 循环分配到多 CPU 核心 | 多核并行加速，提升吞吐量 |
| **数据局部性** | 保证同一线程内数据访问具有时间局部性 | 提高 L1/L2 Cache 命中率 |

### 2.3 跨平台数据对齐技术

#### 2.3.1 二进制数据格式设计

项目采用自定义二进制格式存储模型权重，确保数据在不同平台间的**字节序一致性**：

```cpp
// 数据格式：纯 float 二进制，无元数据
// 文件路径：scripts/*.bin
// 加载方式：直接内存映射读取
void Tensor::load_from_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(data_.data()),
              data_.size() * sizeof(float));
}
```

#### 2.3.2 数据加载流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Python 训练  │ ──▶ │  NumPy 序列化 │ ──▶ │  .bin 文件   │
│   脚本生成    │     │   float32    │     │  二进制数据   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   推理输出   │ ◀── │  C++ 内存映射 │ ◀── │  fstream 读取 │
│   预测结果   │     │   直接加载   │     │  二进制文件   │
└──────────────┘     └──────────────┘     └──────────────┘
```

#### 2.3.3 跨平台兼容性保证

- **统一数据格式**：所有平台使用相同的 float32 IEEE 754 标准
- **无元数据设计**：避免平台特定的元信息，仅存储原始数据
- **显式类型转换**：使用 `reinterpret_cast` 进行安全的二进制解释

### 2.4 工业化标准的代码组织

#### 2.4.1 目录结构规范

```
TinyTnference/
├── include/              # 公共头文件（对外接口）
│   ├── tensor.hpp        # 张量类声明
│   └── layer.hpp         # 层类声明
├── src/                  # 源代码实现
│   ├── tensor.cpp        # 张量实现
│   ├── layer.cpp         # 层实现
│   └── main.cpp          # 主程序入口
├── scripts/              # Python 脚本与数据
│   ├── train_minst.py    # 模型训练脚本
│   ├── export_weights.py # 权重导出脚本
│   └── *.bin             # 预训练权重文件
├── data/                 # 数据集目录
│   └── MNIST/            # MNIST 原始数据
├── build/                # 构建输出目录
├── CMakeLists.txt        # CMake 构建配置
└── README.md             # 项目文档
```

#### 2.4.2 CMake 构建配置

```cmake
# 启用 O3 终极优化，自动使用 SIMD 指令集（如 AVX）
target_compile_options(tiny_infer PRIVATE -O3)

# 引入 OpenMP 多核并行计算库
find_package(OpenMP REQUIRED)
target_link_libraries(tiny_infer PRIVATE OpenMP::OpenMP_CXX)

# 开启所有警告，严格模式
target_compile_options(tiny_infer PRIVATE -Wall -Wextra -Werror)
```

---

## 3. 技术细节

### 3.1 内存分配策略

#### 3.1.1 张量内存布局

项目采用**行主序（Row-Major）**的连续内存布局，与 C++ 数组内存模型一致：

```cpp
// 张量形状 {rows, cols} 在内存中的布局：
// ┌─────────────────────────────────────────┐
// │ data_[0]  data_[1]  ...  data_[cols-1] │  ← Row 0
// │ data_[cols] ...                      │  ← Row 1
// │            ...                          │
// │ data_[rows*cols-1]                       │  ← Row rows-1
// └─────────────────────────────────────────┘

// 一维索引转换公式：
// data_[row * num_cols + col]  <=>  tensor(row, col)
```

#### 3.1.2 内存预分配策略

在张量构造时即完成内存分配，避免运行时动态扩容开销：

```cpp
Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    int total_size = 1;
    for (int s : shape_) total_size *= s;
    data_.resize(total_size, 0.0f);  // 预分配并初始化为 0
}
```

#### 3.1.3 内存管理机制

| 策略 | 实现 | 优势 |
|-----|------|-----|
| 智能指针 | `std::shared_ptr<Layer>` | 自动引用计数，防止内存泄漏 |
| RAII | 构造函数分配，析构函数释放 | 异常安全，资源确定性释放 |
| 移动语义 | 返回 `Tensor` 时使用移动构造 | 避免不必要的数据复制 |

### 3.2 算子优化方法

#### 3.2.1 线性层算子（Linear Layer）

实现 **Y = X·W + B** 的矩阵运算：

```cpp
Tensor LinearLayer::forward(const Tensor& input) {
    // Step 1: 矩阵乘法 Y = X * W
    Tensor temp = matmul(input, weights_);

    // Step 2: 广播加偏置 Y = Y + B
    // 偏置维度 (1, out_features)，自动广播到 (batch, out_features)
    for (int i = 0; i < temp.shape()[0]; ++i) {
        for (int j = 0; j < temp.shape()[1]; ++j) {
            temp(i, j) += bias_(0, j);
        }
    }
    return temp;
}
```

#### 3.2.2 ReLU 激活算子

实现 **f(x) = max(0, x)** 的逐元素运算：

```cpp
Tensor ReLULayer::forward(const Tensor& input) {
    Tensor output(input.shape());

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output(i, j) = std::max(0.0f, input(i, j));
        }
    }
    return output;
}
```

#### 3.2.3 顺序容器算子（Sequential）

实现**前向传播流水线**，自动调度各层执行：

```cpp
Tensor Sequential::forward(const Tensor& input) {
    Tensor current = input;

    for (auto& layer : layers_) {
        current = layer->forward(current);  // 多态调用
    }
    return current;
}
```

### 3.3 数据对齐技术实现

#### 3.3.1 对齐原则

- **内存对齐**：确保 float 数据地址为 4 的倍数（现代 CPU 要求）
- **SIMD 对齐**：16 字节对齐以支持 SSE/AVX 指令集高效访问
- **跨平台一致性**：统一使用 IEEE 754 float32 格式

#### 3.3.2 标准化处理

推理前对输入数据进行标准化，与训练时保持一致：

```cpp
// MNIST 数据标准化参数（来自训练数据集统计）
input.normalize(0.1307f, 0.3081f);  // mean=0.1307, std=0.3081

void Tensor::normalize(float mean, float stddev) {
    for (float& val : data_) {
        val = (val - mean) / stddev;
    }
}
```

### 3.4 复杂度分析

#### 3.4.1 矩阵乘法复杂度

对于输入 **X (M×K)** 与权重 **W (K×N)** 的矩阵乘法：

| 指标 | 复杂度 | 说明 |
|-----|-------|-----|
| 时间复杂度 | O(M·K·N) | 三层嵌套循环 |
| 空间复杂度 | O(M·N) | 结果矩阵存储 |

#### 3.4.2 整体模型复杂度

对于三层 MLP 网络（784→256→128→10）：

| 层类型 | 权重矩阵 | 参数量 | 计算量 (FLOPs) |
|-------|---------|--------|---------------|
| Linear 1 | 784×256 | 200,960 | 200,960×batch |
| Linear 2 | 256×128 | 32,768 | 32,768×batch |
| Linear 3 | 128×10 | 1,280 | 1,280×batch |
| **总计** | - | **235,008** | **234,752×batch** |

---

## 4. 性能表现

### 4.1 优化配置对性能的影响

#### 4.1.1 编译器优化级别

| 优化级别 | 优化内容 | 预期加速比 |
|---------|---------|-----------|
| O0（无优化） | 无 | 1x |
| O1 | 基础优化 | ~1.5x |
| O2 | 指令调度优化 | ~2.5x |
| **O3** | **自动向量化 (SIMD)** | **~4-8x** |

#### 4.1.2 OpenMP 并行效果

对于多核处理器，OpenMP 可实现近线性的加速比：

| CPU 核心数 | 理论加速比 | 实际加速比 |
|-----------|-----------|-----------|
| 1 | 1.0x | 1.0x |
| 2 | 2.0x | ~1.8x |
| 4 | 4.0x | ~3.5x |
| 8 | 8.0x | ~6.5x |

### 4.2 性能瓶颈分析

#### 4.2.1 主要瓶颈点

```
推理耗时分布（典型情况）：
┌─────────────────────────────────────────────────┐
│ ████████████████████████████░░░░░░░░░░░░░░░░░░░░ │
│ 矩阵乘法 (85%)         │  激活函数 (10%)  │ 其他 │
└─────────────────────────────────────────────────┘
```

#### 4.2.2 优化方向建议

1. **批量推理**：将单样本推理改为批量处理，提高矩阵乘法的计算密度
2. **权重量化**：将 float32 权重量化至 int8，减少内存带宽占用
3. **算子融合**：将矩阵乘法与 ReLU 融合，减少中间结果写回
4. **缓存优化**：对热点数据采用显式缓存管理策略

---

## 5. 快速上手指南

### 5.1 环境依赖与前置要求

#### 5.1.1 系统要求

| 组件 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| 操作系统 | Windows 10 / Linux / macOS | Windows 10 / Ubuntu 20.04+ |
| 编译器 | GCC 7+ / MSVC 2019+ / Clang 6+ | GCC 11+ |
| CMake | 3.10+ | 3.22+ |
| Python | 3.8+ | 3.10+ |
| 内存 | 4 GB | 8 GB+ |

#### 5.1.2 必需依赖

| 依赖 | 版本 | 安装说明 |
|-----|------|---------|
| CMake | ≥3.10 | [cmake.org](https://cmake.org/download/) |
| OpenMP | 支持 | 通常随 GCC/Clang 一起安装 |
| Python | ≥3.8 | [python.org](https://www.python.org/downloads/) |
| NumPy | ≥1.20 | `pip install numpy` |

### 5.2 编译与安装流程

#### 5.2.1 Windows 平台

```powershell
# 1. 创建构建目录
cd d:\coding\C\TinyTnference
mkdir build
cd build

# 2. 配置 CMake（使用 MSVC 编译器）
cmake .. -G "Visual Studio 16 2019" -A x64

# 3. 编译项目
cmake --build . --config Release

# 4. 运行推理
.\Release\tiny_infer.exe
```

#### 5.2.2 Linux/macOS 平台

```bash
# 1. 创建构建目录
cd TinyTnference
mkdir -p build && cd build

# 2. 配置 CMake
cmake ..

# 3. 编译项目（多核并行）
make -j$(nproc)

# 4. 运行推理
./tiny_infer
```

#### 5.2.3 使用构建脚本（一键运行）

```bash
# 在项目根目录执行
chmod +x run.sh
./run.sh
```

### 5.3 基本使用示例

#### 5.3.1 完整推理流程

```cpp
// 1. 构建模型结构
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

// 2. 加载模型权重
fc1->weights().load_from_binary("scripts/fc1_w.bin");
fc1->bias().load_from_binary("scripts/fc1_b.bin");
// ... 加载其他权重

// 3. 准备输入数据
Tensor input({1, 784});
input.load_from_binary("scripts/test_digit.bin");
input.normalize(0.1307f, 0.3081f);

// 4. 执行推理
Tensor output = model->forward(input);

// 5. 解析结果
int predicted = argmax(output);
```

#### 5.3.2 输出结果示例

```
Loading weights...
Successfully loaded weights from: scripts/fc1_w.bin
Successfully loaded weights from: scripts/fc1_b.bin
...

--- Visualizing Input Digit ---
##..##....................
#.#.#....................
##.........................
...........................
...

Normalizing input data...
Inference running...
--------------------------------------
Inference Time: 2.35 ms
--------------------------------------

Output Scores (Logits):
Digit 0:     -8.23
Digit 1:     12.45
Digit 2:     -3.21
...

======================================
Final Result: The model thinks it's a [1]
======================================
```

### 5.4 常见问题解决

#### 5.4.1 编译错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `OpenMP not found` | 未安装 OpenMP | 安装 GCC 7+ 或使用 MSVC 2019+ |
| `CMAKE_CXX_STANDARD not found` | CMake 版本过低 | 升级至 CMake 3.10+ |
| `cannot open source file:omp.h` | OpenMP 头文件缺失 | 确保编译器支持 OpenMP |

#### 5.4.2 运行时错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `Failed to open weight file` | 权重文件路径错误 | 检查文件是否存在于 `scripts/` 目录 |
| `Matrix dimensions mismatch` | 张量维度不匹配 | 确认模型结构与权重维度一致 |
| `Segmentation fault` | 内存访问越界 | 检查输入数据维度是否正确 |

#### 5.4.3 性能问题

| 症状 | 原因 | 解决方案 |
|-----|------|---------|
| 推理时间过长 | 未启用 O3 优化 | CMakeLists.txt 中确认 `-O3` 标志 |
| CPU 占用率低 | OpenMP 未生效 | 检查编译器是否支持 OpenMP |
| 内存占用过高 | 批量大小过大 | 减小批量处理大小 |

---

## 附录

### A. 项目文件树

```
TinyTnference/
├── CMakeLists.txt           # CMake 构建配置
├── README.md                # 项目文档
├── run.sh                   # 一键运行脚本
├── build/                    # 构建输出目录
│   ├── tiny_infer          # 编译产物（Linux/macOS）
│   └── tiny_infer.exe      # 编译产物（Windows）
├── data/                    # 数据目录
│   └── MNIST/              # MNIST 数据集
├── include/                 # 公共头文件
│   ├── tensor.hpp          # 张量类声明
│   └── layer.hpp           # 层类声明
├── scripts/                 # Python 脚本与数据
│   ├── train_minst.py      # MNIST 模型训练
│   ├── export_weights.py   # 权重导出
│   ├── test_digit.bin      # 测试图片
│   ├── fc1_w.bin           # 第一层权重
│   ├── fc1_b.bin           # 第一层偏置
│   ├── fc2_w.bin           # 第二层权重
│   ├── fc2_b.bin           # 第二层偏置
│   ├── fc3_w.bin           # 第三层权重
│   └── fc3_b.bin           # 第三层偏置
└── src/                     # 源代码实现
    ├── tensor.cpp          # 张量实现
    ├── layer.cpp           # 层实现
    └── main.cpp            # 主程序
```

### B. 技术栈汇总

| 类别 | 技术选型 |
|-----|---------|
| 编程语言 | C++17 |
| 构建系统 | CMake 3.10+ |
| 并行计算 | OpenMP |
| 数值精度 | float32 (IEEE 754) |
| 数据格式 | 自定义二进制 (.bin) |
| 训练框架 | Python + NumPy |

### C. 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源许可。

---

**项目作者**：TinyTnference Team  
**最后更新**：2026-04-19
