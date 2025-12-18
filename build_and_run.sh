#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "======================================"
echo "一键编译并运行 TetgenFEM"
echo "======================================"

./build.sh
./run.sh

