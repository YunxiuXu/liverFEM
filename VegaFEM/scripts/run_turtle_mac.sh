#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLE_DIR="${REPO_ROOT}/examples/turtle"
BIN="${REPO_ROOT}/build/macos-release/utilities/interactiveDeformableSimulator"
CONFIG="turtle.config"

if [[ ! -x "${BIN}" ]]; then
  echo "[vegafem] 找不到可执行文件: ${BIN}"
  echo "[vegafem] 先运行 /usr/bin/arch -x86_64 ${REPO_ROOT}/scripts/build_run_mac.sh"
  exit 1
fi

if [[ ! -f "${EXAMPLE_DIR}/${CONFIG}" ]]; then
  echo "[vegafem] 找不到配置: ${EXAMPLE_DIR}/${CONFIG}"
  exit 1
fi

cd "${EXAMPLE_DIR}"
echo "[vegafem] 运行 turtle 示例..."
/usr/bin/arch -x86_64 "${BIN}" "${CONFIG}"
