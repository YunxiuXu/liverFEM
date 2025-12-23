#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VEGAFEM_DIR="${ROOT_DIR}/VegaFEM"
BUILD_DIR="${VEGAFEM_DIR}/build/macos-release"
BIN="${BUILD_DIR}/utilities/interactiveDeformableSimulator"

log() {
    echo "[vegafem] $*"
}

# 解析配置文件路径
CONFIG_ARG="${1:-liver_HD_Low}"

# 1. 编译
log "正在编译 VegaFEM..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
if [[ "$(uname -m)" == "arm64" ]]; then
    ARCH_FLAG="x86_64"
    CMAKE_PREFIX="/usr/local"
else
    ARCH_FLAG="$(uname -m)"
    CMAKE_PREFIX="/usr/local"
fi
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null
if [[ ! -f "CMakeCache.txt" ]]; then
    cmake ../.. -DCMAKE_BUILD_TYPE=Release -DVEGAFEM_USE_MKL=OFF -DCMAKE_OSX_ARCHITECTURES="${ARCH_FLAG}" -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}"
fi
make -j$(sysctl -n hw.logicalcpu) interactiveDeformableSimulator

# 2. 运行
if [[ -f "${VEGAFEM_DIR}/${CONFIG_ARG}.config" ]]; then
    CONFIG_FILE="${VEGAFEM_DIR}/${CONFIG_ARG}.config"
else
    CONFIG_FILE="${VEGAFEM_DIR}/${CONFIG_ARG}"
fi

CONFIG_DIR="$(cd "$(dirname "${CONFIG_FILE}")" && pwd)"
CONFIG_NAME="$(basename "${CONFIG_FILE}")"

log "启动程序: ${CONFIG_NAME}"
cd "${CONFIG_DIR}"
if [[ "$(uname -m)" == "arm64" && "${ARCH_FLAG}" == "x86_64" ]]; then
    arch -x86_64 "${BIN}" "${CONFIG_NAME}"
else
    "${BIN}" "${CONFIG_NAME}"
fi
