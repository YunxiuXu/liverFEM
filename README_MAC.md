# TetgenFEM - Mac 使用指南

这是 TetgenFEM 项目的 Mac 版本编译和运行指南。

## 系统要求

- macOS 10.14 或更高版本
- Xcode Command Line Tools
- Homebrew 包管理器

## 依赖安装

### 1. 安装 Homebrew（如果还没安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. 安装必要的依赖

```bash
brew install cmake
brew install glfw
brew install glew
brew install libomp  # OpenMP 支持（可选，用于多线程加速）
```

### 3. 安装 Xcode Command Line Tools（如果还没安装）

```bash
xcode-select --install
```

## 编译项目

在项目根目录运行：

```bash
./build.sh
```

这个脚本会：
1. 检查必要的依赖是否安装
2. 创建 build 目录
3. 运行 CMake 配置
4. 编译项目

## 运行程序

编译成功后，在项目根目录运行：

```bash
./run.sh
```

或者手动运行：

```bash
cd TetgenFEM
../out/build/TetgenFEM
```

**注意**: 程序必须在 `TetgenFEM` 目录下运行，因为它需要读取该目录下的 `parameters.txt` 和 `models/` 资源文件夹。

## 配置参数

编辑 `TetgenFEM/parameters.txt` 来修改仿真参数：

```
youngs=1000000          # 杨氏模量
poisson=0.28            # 泊松比
density=1000            # 密度
groupNumX=5             # X 方向分组数
groupNumY=8             # Y 方向分组数
groupNumZ=4             # Z 方向分组数
timeStep=0.01           # 时间步长
dampingConst=20.0       # 阻尼系数
Gravity=-10.0           # 重力加速度
modelDir=models         # 资源文件夹（STL / node / ele 等都放这里）
stlFile=ring.stl        # STL 文件名（会自动拼到 modelDir 下）
tetgenArgs=pq2a0.0005   # TetGen 参数
```

## Mac 特定的修改

与 Windows 版本相比，Mac 版本进行了以下修改：

1. **Windows.h 头文件**: 使用条件编译 `#ifdef _WIN32` 只在 Windows 上包含
2. **字体渲染**: Windows 特定的 `wglUseFontBitmaps` 在 Mac 上被禁用
3. **OpenGL 警告**: 添加了 `-DGL_SILENCE_DEPRECATION` 来抑制 OpenGL 弃用警告
4. **编译器警告**: 禁用了 Eigen 库产生的 `-Wnan-infinity-disabled` 警告

## 交互操作

- **鼠标左键拖拽**: 旋转视图
- **鼠标滚轮**: 缩放视图
- **W/S/A/D 键**: 可能用于特殊操作（取决于代码实现）
- **C 键**: 保存当前顶点坐标到文件

## Ultraleap / Leap Motion 输入（右手五指指尖）

项目支持使用 Ultraleap Gemini (v5.x) 的 LeapC 接口，把 **右手五个指尖** 映射到仿真里的 5 个 agent sphere（proxy/device）。

1. 确保 `Ultraleap/LeapC.h` 和 `Ultraleap/libLeapC.dylib` 存在（本仓库已包含）。
2. macOS 可能需要在“系统设置 → 隐私与安全性”中给你的终端或 App 授予 **输入监控 (Input Monitoring)** 权限，然后重启程序。
3. 运行程序后：
   - 按 **H** 打开/关闭 agent sphere，并在控制台打印快捷键说明
   - 按 **B** 开关 Leap 输入
   - 按 **R** 重新居中（用当前食指指尖位置作为零点）
   - 按 **[** / **]** 调整映射灵敏度（gain）

映射相关参数在 `TetgenFEM/parameters.txt`：
- `leap_enabled`：启动时是否默认开启 Leap
- `leap_workspaceXmm/Ymm/Zmm`：Leap 的工作空间尺寸（mm，用于把指尖位移映射到模型 bbox）
- `leap_worldMargin`：bbox 映射/夹取的额外边界比例（避免手指“满世界飞”）
- `leap_gain`：整体映射倍率（感觉活动范围太小就调大）
- `leap_yOffsetBboxFrac`：映射后的 Y 轴整体偏移（bbox 高度的比例；负数=整体更低）
- `leap_fingerSpreadGain`：五指之间的“展开”倍率（只放大指尖之间的相对距离，不影响整体移动范围；看不到其他手指时调大）
- `leap_smoothingTime`：指数平滑时间常数（秒，抑制抖动）
- `leap_flipX/Y/Z`：轴向翻转（如果 Z 方向反了，把 `leap_flipZ` 设为 `true`）

## 常见问题

### 编译时找不到 OpenGL

确保安装了 Xcode Command Line Tools:
```bash
xcode-select --install
```

### 找不到 GLFW 或 GLEW

使用 Homebrew 安装:
```bash
brew install glfw glew
```

### 链接错误: 找不到 tetgen.lib

tetgen.lib 是 Windows 库文件。如果需要 Mac 版本，需要：
1. 下载 TetGen 源码
2. 编译成 .a 或 .dylib 文件
3. 更新 CMakeLists.txt 中的链接路径

### OpenMP 警告

如果看到 OpenMP 相关警告，可以安装 libomp:
```bash
brew install libomp
```

然后重新编译项目。

## 性能优化

- 项目已启用 `-O3` 优化
- 支持 OpenMP 多线程并行（如果可用）
- 可以通过修改 `parameters.txt` 调整分组数来平衡性能和精度

## 技术支持

如有问题，请参考原项目的 README.md 或联系项目维护者。

## 更新日志

- 2025-12-11: 添加 Mac 支持，修复跨平台兼容性问题
