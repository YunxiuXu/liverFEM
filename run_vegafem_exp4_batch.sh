#!/usr/bin/env bash
# 这是一个批处理脚本，用于依次运行 VegaFEM 的所有性能测试点
# 它由 VegaFEM GUI 里的 Experiment 4 按钮点击触发

ROOT_DIR="/Users/yunxiuxu/Documents/tetfemcpp"
VEGAFEM_DIR="${ROOT_DIR}/VegaFEM"
BIN="${VEGAFEM_DIR}/build/macos-release/utilities/interactiveDeformableSimulator"

export VEGAFEM_AUTO_EXP4=1

log() {
    echo "[exp4-batch] $*"
}

run_point() {
    local conf="$1"
    if [[ -f "${VEGAFEM_DIR}/${conf}.config" ]]; then
        log "Starting performance test for: ${conf}"
        cd "${VEGAFEM_DIR}"
        # 直接运行二进制文件，跳过编译步骤以节省时间并避免冲突
        if [[ "$(uname -m)" == "arm64" ]]; then
            arch -x86_64 "${BIN}" "${conf}.config"
        else
            "${BIN}" "${conf}.config"
        fi
    else
        log "Warning: Config not found: ${conf}.config"
    fi
}

log "=========================================="
log "Starting full Experiment 4 batch run..."
log "=========================================="

# 依次运行不同规模的网格
run_point "liver_target5000"
run_point "liver_target20000"
run_point "liver_target50000"

log "=========================================="
log "Full Experiment 4 batch run completed."
log "=========================================="
