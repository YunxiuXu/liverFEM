#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBD_DIR="${ROOT_DIR}/PositionBasedDynamics"
BUILD_DIR="${ROOT_DIR}/build/pbd_user"
PBD_CACHE_BUILD_DIR="${ROOT_DIR}/build/pbd"

if [[ ! -d "${PBD_DIR}" ]]; then
  echo "Missing PositionBasedDynamics at: ${PBD_DIR}" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found. Install it first (e.g. via Xcode Command Line Tools / Homebrew)." >&2
  exit 1
fi

SCENE_PATH="${1:-"${PBD_DIR}/data/scenes/CarScene.json"}"
if [[ ! -f "${SCENE_PATH}" ]]; then
  echo "Scene file not found: ${SCENE_PATH}" >&2
  echo "Usage: $0 [scene.json]" >&2
  exit 1
fi

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 8)"

PBD_LOCAL_DEPS_CMAKE_ARGS=""
DISCREGRID_LOCAL_REPO="${PBD_CACHE_BUILD_DIR}/extern/Discregrid/src/Ext_Discregrid"
GENPARAMS_LOCAL_REPO="${PBD_CACHE_BUILD_DIR}/GenericParameters/src/Ext_GenericParameters"
if [[ -d "${DISCREGRID_LOCAL_REPO}/.git" && -d "${GENPARAMS_LOCAL_REPO}/.git" ]]; then
  PBD_LOCAL_DEPS_CMAKE_ARGS="-DPBD_DISCREGRID_REPO=${DISCREGRID_LOCAL_REPO} -DPBD_GENERICPARAMETERS_REPO=${GENPARAMS_LOCAL_REPO}"
else
  echo "Note: Discregrid/GenericParameters local mirrors not found under ${PBD_CACHE_BUILD_DIR}." >&2
  echo "      First build may require network access (CMake ExternalProject will clone from GitHub)." >&2
fi

cmake -S "${PBD_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_PYTHON_BINDINGS=Off \
  -DPBD_RUNTIME_OUTPUT_DIRECTORY="${BUILD_DIR}/bin" \
  ${PBD_LOCAL_DEPS_CMAKE_ARGS:-}

cmake --build "${BUILD_DIR}" -j "${JOBS}"

pushd "${BUILD_DIR}/bin" >/dev/null
./SceneLoaderDemo "${SCENE_PATH}"
popd >/dev/null
