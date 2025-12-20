#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VEGAFEM_DIR="${ROOT_DIR}/VegaFEM"
BUILD_DIR="${VEGAFEM_DIR}/build/macos-release"
BIN="${BUILD_DIR}/utilities/interactiveDeformableSimulator"

log() {
    echo "[vegafem] $*"
}

# 显示帮助信息
show_help() {
    log "用法: $0 [配置文件名称或路径]"
    log ""
    log "示例:"
    log "  $0 cubeLong1300              # 使用 cubeLong1300.config"
    log "  $0 liver_HD_Low              # 使用 liver_HD_Low.config"
    log "  $0 cubeLong1300.config        # 完整文件名"
    log "  $0 ${VEGAFEM_DIR}/cubeLong1300.config  # 完整路径"
    log ""
    log "可用的配置文件（在 ${VEGAFEM_DIR} 目录下）:"
    find "${VEGAFEM_DIR}" -maxdepth 1 -name "*.config" -type f 2>/dev/null | while read -r f; do
        log "  - $(basename "$f")"
    done || log "  (无法列出配置文件)"
    exit 0
}

# 处理命令行参数
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "help" ]]; then
    show_help
fi

# 解析配置文件路径
CONFIG_ARG="${1:-cubeLong1300}"

# 如果参数是完整路径且文件存在，直接使用
if [[ -f "${CONFIG_ARG}" ]]; then
    CONFIG_FILE="${CONFIG_ARG}"
# 如果参数包含路径分隔符，尝试作为相对路径
elif [[ "${CONFIG_ARG}" == */* ]]; then
    CONFIG_FILE="${CONFIG_ARG}"
# 否则，在 VegaFEM 目录下查找
else
    # 移除可能的 .config 后缀（如果用户输入了）
    CONFIG_BASE="${CONFIG_ARG%.config}"
    # 尝试查找配置文件
    if [[ -f "${VEGAFEM_DIR}/${CONFIG_BASE}.config" ]]; then
        CONFIG_FILE="${VEGAFEM_DIR}/${CONFIG_BASE}.config"
    elif [[ -f "${VEGAFEM_DIR}/${CONFIG_BASE}" ]]; then
        CONFIG_FILE="${VEGAFEM_DIR}/${CONFIG_BASE}"
    else
        CONFIG_FILE="${VEGAFEM_DIR}/${CONFIG_BASE}.config"
    fi
fi

# 检查配置文件是否存在
if [[ ! -f "${CONFIG_FILE}" ]]; then
    log "错误: 配置文件不存在: ${CONFIG_FILE}"
    log ""
    show_help
fi

# 确保在 VegaFEM 目录下运行（配置文件中的相对路径需要这个）
CONFIG_DIR="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)"
CONFIG_NAME="$(basename "${CONFIG_FILE}")"

log "=========================================="
log "VegaFEM 构建和运行脚本"
log "=========================================="
log "配置文件: ${CONFIG_FILE}"
log "工作目录: ${CONFIG_DIR}"
log "=========================================="

# 检查依赖
if ! command -v cmake &> /dev/null; then
    log "错误: 未找到 cmake"
    exit 1
fi

# 创建构建目录
mkdir -p "${BUILD_DIR}"

# 配置 CMake（强制使用 x86_64 架构，使用 /usr/local 的库以匹配 Rosetta）
log "配置 CMake..."
cd "${BUILD_DIR}"

# 检测是否需要 Rosetta（在 Apple Silicon 上）
if [[ "$(uname -m)" == "arm64" ]]; then
    ARCH_FLAG="x86_64"
    # 在 Apple Silicon 上，x86_64 库通常在 /usr/local（Rosetta Homebrew）
    CMAKE_PREFIX="/usr/local"
    log "检测到 Apple Silicon，将编译 x86_64 版本（使用 Rosetta）"
else
    ARCH_FLAG="$(uname -m)"
    CMAKE_PREFIX="/usr/local"
fi

# 清理旧的 CMake 缓存（如果架构不匹配或首次构建）
if [[ -f "CMakeCache.txt" ]]; then
    OLD_ARCH=$(grep "CMAKE_OSX_ARCHITECTURES" CMakeCache.txt 2>/dev/null | cut -d'=' -f2 || echo "")
    if [[ "${OLD_ARCH}" != "${ARCH_FLAG}" ]]; then
        log "检测到架构变更 (${OLD_ARCH} -> ${ARCH_FLAG})，清理构建目录..."
        cd "${VEGAFEM_DIR}"
        rm -rf "${BUILD_DIR}"
        mkdir -p "${BUILD_DIR}"
        cd "${BUILD_DIR}"
    fi
else
    # 首次构建，清理可能存在的旧依赖
    log "首次构建，清理旧的依赖缓存..."
    rm -rf "${BUILD_DIR}/_deps"
fi

# 配置 CMake
# 设置环境变量以避免 git 仓库检查问题
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null

if [[ ! -f "CMakeCache.txt" ]] || ! cmake ../.. \
    -DCMAKE_BUILD_TYPE=Release \
    -DVEGAFEM_USE_MKL=OFF \
    -DCMAKE_OSX_ARCHITECTURES="${ARCH_FLAG}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}" \
    -DFETCHCONTENT_QUIET=OFF \
    -DFETCHCONTENT_FULLY_DISCONNECTED=OFF \
    &>/dev/null; then
    log "重新配置 CMake..."
    cmake ../.. \
        -DCMAKE_BUILD_TYPE=Release \
        -DVEGAFEM_USE_MKL=OFF \
        -DCMAKE_OSX_ARCHITECTURES="${ARCH_FLAG}" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}" \
        -DFETCHCONTENT_QUIET=OFF \
        -DFETCHCONTENT_FULLY_DISCONNECTED=OFF
fi

# 编译
log "编译 interactiveDeformableSimulator..."
make -j$(sysctl -n hw.logicalcpu) interactiveDeformableSimulator

# 检查二进制文件是否存在
if [[ ! -x "${BIN}" ]]; then
    log "错误: 编译失败，未找到可执行文件: ${BIN}"
    exit 1
fi

log "编译成功: ${BIN}"

# 运行程序（在配置文件所在目录下运行，以便相对路径正确解析）
log "运行程序..."
log "=========================================="
cd "${CONFIG_DIR}"

# 如果在 Apple Silicon 上编译了 x86_64 版本，需要使用 Rosetta 运行
if [[ "$(uname -m)" == "arm64" && "${ARCH_FLAG}" == "x86_64" ]]; then
    log "使用 Rosetta (x86_64) 运行..."
    arch -x86_64 "${BIN}" "${CONFIG_NAME}"
else
    "${BIN}" "${CONFIG_NAME}"
fi
