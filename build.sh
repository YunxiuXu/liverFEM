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

# 检查 libomp（OpenMP 支持）
if ! brew list libomp &> /dev/null; then
    echo "警告: 未找到 libomp，尝试安装以启用 OpenMP..."
    brew install libomp
fi

# 优先使用 Homebrew 的 llvm/clang 以获得完整 OpenMP 支持
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

# 检查 GLEW
if ! brew list glew &> /dev/null; then
    echo "警告: 未找到 glew，尝试安装..."
    brew install glew
fi

# 创建 build 目录（放到 out/ 下，避免把构建产物混进仓库）
BUILD_DIR="out/build"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
# 清理旧的缓存以便切换编译器
rm -f CMakeCache.txt

# 运行 CMake
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

# 编译
echo "编译..."
make -j$(sysctl -n hw.ncpu)

if [ $? -eq 0 ]; then
    echo "======================================"
    echo "编译成功！"
    echo "可执行文件位于: ${BUILD_DIR}/TetgenFEM"
    echo "======================================"
else
    echo "======================================"
    echo "编译失败！"
    echo "======================================"
    exit 1
fi
