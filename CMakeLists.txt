// 最低的版本要求
cmake_minimum_required(VERSION 3.10)

//定义项目名称和版本
project(TinyTnference VERSION 1.0 LANGUAGES CXX)

//指定C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

// 告诉编译器在哪里可以找到头文件
include_directories(include)

//查找当前目录下的所有源文件
file(GLOB SOURCES "src/*.cpp")

//生成可执行文件
add_executable(TinyTnference ${SOURCES})

//开启所有警告
if (MSVC)
    target_compile_options(TinyTnference PRIVATE /W4 /WX)
else()
    target_compile_options(TinyTnference PRIVATE -Wall -Wextra -Werror)
endif()
