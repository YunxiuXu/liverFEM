#!/bin/bash

echo "======================================"
echo "安装 TetgenFEM Mac 依赖"
echo "======================================"

# 检查 Homebrew 是否安装
if ! command -v brew &> /dev/null; then
    echo "Homebrew 未安装，正在安装..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✓ Homebrew 已安装"
fi

# 更新 Homebrew
echo "更新 Homebrew..."
brew update

# 安装 CMake
echo "检查 CMake..."
if ! command -v cmake &> /dev/null; then
    echo "安装 CMake..."
    brew install cmake
else
    echo "✓ CMake 已安装"
fi

# 安装 GLFW
echo "检查 GLFW..."
if ! brew list glfw &> /dev/null; then
    echo "安装 GLFW..."
    brew install glfw
else
    echo "✓ GLFW 已安装"
fi

# 安装 GLEW
echo "检查 GLEW..."
if ! brew list glew &> /dev/null; then
    echo "安装 GLEW..."
    brew install glew
else
    echo "✓ GLEW 已安装"
fi

# 安装 OpenMP（可选）
echo "检查 OpenMP..."
if ! brew list libomp &> /dev/null; then
    echo "安装 OpenMP（用于多线程加速）..."
    brew install libomp
else
    echo "✓ OpenMP 已安装"
fi

# 检查 Xcode Command Line Tools
echo "检查 Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
    echo "安装 Xcode Command Line Tools..."
    xcode-select --install
else
    echo "✓ Xcode Command Line Tools 已安装"
fi

echo "======================================"
echo "依赖安装完成！"
echo "======================================"
echo ""
echo "下一步:"
echo "1. 运行 ./build.sh 编译项目"
echo "2. 运行 ./run.sh 运行程序"
echo ""
