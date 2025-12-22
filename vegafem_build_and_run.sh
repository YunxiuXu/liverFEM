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
    log "用法: $0 [配置文件名称 | all]"
    log ""
    log "选项:"
    log "  all                          # 一键运行实验 4 的所有规模 (liver, liver_HD_Low, liver_HD_High)"
    log "  [config_name]                # 运行指定的配置文件"
    log ""
    log "示例:"
    log "  $0 all                       # 全自动运行实验 4"
    log "  $0 liver_HD_Low              # 运行 20k 网格"
    exit 0
}

# 处理命令行参数
ARG="${1:-all}"
if [[ "${ARG}" == "-h" || "${ARG}" == "--help" ]]; then
    show_help
fi

# 1. 编译逻辑 (只需执行一次)
log "=========================================="
log "Step 1: 检查依赖并编译 VegaFEM"
log "=========================================="

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

if [[ ! -f "CMakeCache.txt" ]] || ! cmake ../.. \
    -DCMAKE_BUILD_TYPE=Release \
    -DVEGAFEM_USE_MKL=OFF \
    -DCMAKE_OSX_ARCHITECTURES="${ARCH_FLAG}" \
    -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}" \
    &>/dev/null; then
    cmake ../.. -DCMAKE_BUILD_TYPE=Release -DVEGAFEM_USE_MKL=OFF -DCMAKE_OSX_ARCHITECTURES="${ARCH_FLAG}" -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX}"
fi

make -j$(sysctl -n hw.logicalcpu) interactiveDeformableSimulator

if [[ ! -x "${BIN}" ]]; then
    log "错误: 编译失败"
    exit 1
fi

# 2. 运行逻辑函数
run_config() {
    local conf_name="$1"
    local auto_exp4="${2:-0}"
    
    # 查找配置文件
    local c_file=""
    if [[ -f "${VEGAFEM_DIR}/${conf_name}.config" ]]; then
        c_file="${VEGAFEM_DIR}/${conf_name}.config"
    elif [[ -f "${VEGAFEM_DIR}/${conf_name}" ]]; then
        c_file="${VEGAFEM_DIR}/${conf_name}"
    else
        c_file="${VEGAFEM_DIR}/${conf_name}.config"
    fi
    
    if [[ ! -f "${c_file}" ]]; then
        log "警告: 配置文件不存在: ${c_file}，跳过"
        return
    fi
    
    local c_dir="$(cd "$(dirname "${c_file}")" && pwd)"
    local b_name="$(basename "${c_file}")"
    
    log "------------------------------------------"
    log "正在运行: ${b_name} (AutoExp4=${auto_exp4})"
    log "------------------------------------------"
    
    cd "${c_dir}"
    if [[ "${auto_exp4}" == "1" ]]; then
        export VEGAFEM_AUTO_EXP4=1
    else
        unset VEGAFEM_AUTO_EXP4
    fi
    
    if [[ "$(uname -m)" == "arm64" && "${ARCH_FLAG}" == "x86_64" ]]; then
        arch -x86_64 "${BIN}" "${b_name}"
    else
        "${BIN}" "${b_name}"
    fi
}

# 3. 执行运行任务
if [[ "${ARG}" == "all" ]]; then
    log "=========================================="
    log "Step 2: 一键执行全量实验 4"
    log "=========================================="
    run_config "liver" "1"
    run_config "liver_HD_Low" "1"
    run_config "liver_HD_High" "1"
    log "=========================================="
    log "所有 VegaFEM 性能测试已完成"
    log "=========================================="
else
    run_config "${ARG}" "0"
fi
