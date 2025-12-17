# TetgenFEM - Mac 快速开始

## 一键安装和运行

### 步骤 1: 安装依赖（只需要运行一次）

```bash
cd /Users/yunxiuxu/Documents/tetfemcpp
./install_deps_mac.sh
```

这将自动安装所有必需的依赖：
- Homebrew（如果未安装）
- CMake
- GLFW
- GLEW
- OpenMP
- Xcode Command Line Tools

### 步骤 2: 编译项目

```bash
./build.sh
```

### 步骤 3: 运行程序

```bash
./run.sh
```

## 完整命令（一次性执行）

如果依赖已经安装，直接运行：

```bash
cd /Users/yunxiuxu/Documents/tetfemcpp
./build.sh && ./run.sh
```

## 重新编译

如果修改了代码，只需要重新编译：

```bash
cd out/build
make -j$(sysctl -n hw.ncpu)
cd ..
./run.sh
```

## 清理构建

如果遇到编译问题，可以清理后重新构建：

```bash
rm -rf build
./build.sh
```

## 项目结构

```
tetfemcpp/
├── TetgenFEM/              # 源代码目录
│   ├── main.cpp
│   ├── parameters.txt      # 配置文件
│   ├── models/            # 模型资源（STL / node / ele 等）
│   └── ...
├── out/build/              # 编译输出目录
│   └── TetgenFEM           # 可执行文件
├── CMakeLists.txt         # CMake 配置文件
├── build.sh               # 编译脚本
├── run.sh                 # 运行脚本
└── install_deps_mac.sh    # 依赖安装脚本
```

## 常见问题解决

### 问题: 编译时找不到 OpenGL

**解决方案:**
```bash
xcode-select --install
```

### 问题: 找不到 glfw 或 glew

**解决方案:**
```bash
brew install glfw glew
```

### 问题: OpenMP 警告

**解决方案:**
```bash
brew install libomp
```

### 问题: 程序运行时找不到文件

**解决方案:** 确保从项目根目录运行 `./run.sh`，不要直接运行 `out/build/TetgenFEM`

## 修改参数

编辑 `TetgenFEM/parameters.txt` 文件来调整仿真参数。常用参数：

- `youngs`: 杨氏模量（材料刚度）
- `poisson`: 泊松比
- `density`: 密度
- `groupNumX/Y/Z`: 分组数（影响性能和精度）
- `stlFile`: 要加载的 STL 模型路径
- `Gravity`: 重力加速度

修改参数后，不需要重新编译，直接运行 `./run.sh` 即可。

## 性能提示

1. **调整分组数**: 增加 `groupNumX/Y/Z` 可以提高并行度，但也会增加计算开销
2. **使用 OpenMP**: 确保安装了 `libomp` 以启用多线程加速
3. **模型复杂度**: 使用 `tetgenArgs` 参数控制网格密度，如 `pq2a0.0005`

## 已知限制

1. **文字渲染**: Mac 版本暂不支持文字渲染（Windows 特定功能）
2. **OpenGL 警告**: macOS 10.14+ 已弃用 OpenGL，但程序仍可正常运行

## 更多信息

详细文档请参考 `README_MAC.md`
