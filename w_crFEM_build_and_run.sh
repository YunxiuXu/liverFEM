#!/bin/bash

# 获取脚本所在目录的绝对路径
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/out/build"

echo "======================================"
echo "编译 TetgenFEM"
echo "======================================"

# 检查依赖
echo "检查依赖..."

if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake，请安装: brew install cmake"
    exit 1
fi

if ! brew list glfw &> /dev/null; then
    echo "警告: 未找到 glfw，尝试安装..."
    brew install glfw
fi

if ! brew list libomp &> /dev/null; then
    echo "警告: 未找到 libomp，尝试安装以启用 OpenMP..."
    brew install libomp
fi

if brew list llvm &> /dev/null; then
    export PATH="/opt/homebrew/opt/llvm/bin:${PATH}"
    export CC="/opt/homebrew/opt/llvm/bin/clang"
    export CXX="/opt/homebrew/opt/llvm/bin/clang++"
    export LDFLAGS="-L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib/unwind -lunwind ${LDFLAGS}"
    export CPPFLAGS="-I/opt/homebrew/opt/llvm/include ${CPPFLAGS}"
    export CMAKE_PREFIX_PATH="/opt/homebrew/opt/llvm:${CMAKE_PREFIX_PATH}"
else
    echo "警告: 未找到 llvm（Homebrew 版本）。OpenMP 可能无法在系统 clang 上启用。可通过 'brew install llvm' 安装。"
fi

if ! brew list glew &> /dev/null; then
    echo "警告: 未找到 glew，尝试安装..."
    brew install glew
fi

# 开始构建
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
rm -f CMakeCache.txt

echo "运行 CMake..."
cmake ../.. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_EXE_LINKER_FLAGS="-stdlib=libc++ -L/opt/homebrew/opt/llvm/lib/c++ -L/opt/homebrew/opt/llvm/lib/unwind -lunwind" \
    -DOpenMP_C_INCLUDE_DIR=/opt/homebrew/opt/libomp/include \
    -DOpenMP_CXX_INCLUDE_DIR=/opt/homebrew/opt/libomp/include \
    -DOpenMP_C_LIB_NAMES=omp \
    -DOpenMP_CXX_LIB_NAMES=omp \
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp" \
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
    -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib

echo "编译..."
make -j$(sysctl -n hw.ncpu)

if [ $? -eq 0 ]; then
    echo "======================================"
    echo "编译成功！"
    echo "======================================"
else
    echo "======================================"
    echo "编译失败！"
    echo "======================================"
    exit 1
fi

echo ""
echo "======================================"
echo "运行 TetgenFEM 仿真程序"
echo "======================================"

# 切换到项目运行目录
cd "${ROOT_DIR}/TetgenFEM"

# 运行程序
if [ -f "${BUILD_DIR}/TetgenFEM" ]; then
    "${BUILD_DIR}/TetgenFEM"
else
    echo "错误: 未找到可执行文件 ${BUILD_DIR}/TetgenFEM"
    exit 1
fi

echo "======================================"
echo "程序已退出"
echo "======================================"
