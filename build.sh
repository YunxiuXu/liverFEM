#!/bin/bash

echo "======================================"
echo "编译 TetgenFEM"
echo "======================================"

# 检查是否安装了必要的依赖
echo "检查依赖..."

# 检查 CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake，请安装: brew install cmake"
    exit 1
fi

# 检查 GLFW
if ! brew list glfw &> /dev/null; then
    echo "警告: 未找到 glfw，尝试安装..."
    brew install glfw
fi

# 检查 GLEW
if ! brew list glew &> /dev/null; then
    echo "警告: 未找到 glew，尝试安装..."
    brew install glew
fi

# 创建 build 目录
mkdir -p build
cd build

# 运行 CMake
echo "运行 CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "编译..."
make -j$(sysctl -n hw.ncpu)

if [ $? -eq 0 ]; then
    echo "======================================"
    echo "编译成功！"
    echo "可执行文件位于: build/TetgenFEM"
    echo "======================================"
else
    echo "======================================"
    echo "编译失败！"
    echo "======================================"
    exit 1
fi
