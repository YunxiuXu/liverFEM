#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VEGAFEM_DIR="${ROOT_DIR}/VegaFEM"
BUILD_DIR="${VEGAFEM_DIR}/build/macos-release"
BIN="${BUILD_DIR}/utilities/interactiveDeformableSimulator"

# 默认配置文件（可以覆盖）
CONFIG_FILE="${1:-${VEGAFEM_DIR}/cubeLong1300.config}"

log() {
    echo "[vegafem] $*"
}

# 检查配置文件是否存在
if [[ ! -f "${CONFIG_FILE}" ]]; then
    log "错误: 配置文件不存在: ${CONFIG_FILE}"
    log "用法: $0 [配置文件路径]"
    log "示例: $0 ${VEGAFEM_DIR}/cubeLong1300.config"
    exit 1
fi

# 确保在 VegaFEM 目录下运行（配置文件中的相对路径需要这个）
CONFIG_DIR="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)"
CONFIG_NAME="$(basename "${CONFIG_FILE}")"

log "配置文件: ${CONFIG_FILE}"
log "工作目录: ${CONFIG_DIR}"

# 检查依赖
if ! command -v cmake &> /dev/null; then
    log "错误: 未找到 cmake"
    exit 1
fi

# 创建构建目录
mkdir -p "${BUILD_DIR}"

# 配置 CMake（使用 arm64，优先使用 /opt/homebrew 的库）
log "配置 CMake..."
cd "${BUILD_DIR}"
if [[ ! -f "CMakeCache.txt" ]] || ! cmake ../.. -DCMAKE_BUILD_TYPE=Release -DVEGAFEM_USE_MKL=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_PREFIX_PATH=/opt/homebrew &>/dev/null; then
    log "重新配置 CMake..."
    cmake ../.. \
        -DCMAKE_BUILD_TYPE=Release \
        -DVEGAFEM_USE_MKL=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_PREFIX_PATH=/opt/homebrew
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
"${BIN}" "${CONFIG_NAME}"
