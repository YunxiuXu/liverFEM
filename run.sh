#!/bin/bash

echo "======================================"
echo "运行 TetgenFEM 仿真程序"
echo "======================================"

# 检查可执行文件是否存在
if [ ! -f "build/TetgenFEM" ]; then
    echo "错误: 未找到可执行文件 build/TetgenFEM"
    echo "请先运行 ./build.sh 编译项目"
    exit 1
fi

# 切换到 TetgenFEM 目录（因为程序需要读取 parameters.txt 和 stl 文件）
cd TetgenFEM

# 运行程序
../build/TetgenFEM

echo "======================================"
echo "程序已退出"
echo "======================================"
