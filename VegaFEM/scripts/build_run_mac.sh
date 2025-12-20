#!/usr/bin/env bash
set -euo pipefail

log() { echo "[vegafem] $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/macos-release}"
TARGET_CONFIG="${TARGET_CONFIG:-${REPO_ROOT}/examples/turtle/turtle.config}"
TARGET_EXE_NAME="${TARGET_EXE_NAME:-interactiveDeformableSimulator}"

# Force x86_64 build to match /usr/local Homebrew bottles (Rosetta on Apple Silicon).
if [[ "$(uname -m)" == "arm64" && "${VEGAFEM_ROSETTA_REEXEC:-0}" != "1" ]]; then
  if ! /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
    log "Rosetta 2 is required. Install it with:"
    log "  softwareupdate --install-rosetta --agree-to-license"
    exit 1
  fi
  export VEGAFEM_ROSETTA_REEXEC=1
  exec /usr/bin/arch -x86_64 env VEGAFEM_ROSETTA_REEXEC=1 "$0" "$@"
fi

if [[ -x "/usr/local/bin/brew" ]]; then
  BREW_CMD=("/usr/local/bin/brew")
else
  BREW_CMD=("/usr/bin/env" "brew")
fi

ensure_formula() {
  local pkg="$1"
  if ! "${BREW_CMD[@]}" list --versions "${pkg}" >/dev/null 2>&1; then
    log "Installing ${pkg}..."
    "${BREW_CMD[@]}" install "${pkg}"
  else
    log "Found ${pkg} ($("${BREW_CMD[@]}" list --versions "${pkg}" | tr -s ' '))"
  fi
}

log "Checking Homebrew dependencies..."
ensure_formula tbb
ensure_formula arpack
ensure_formula cgal
ensure_formula eigen
ensure_formula glui
ensure_formula glew
ensure_formula wxwidgets
ensure_formula pkgconf

find_cmake_dir() {
  local prefix="$1"
  local name="$2"
  if [[ -d "${prefix}" ]]; then
    find "${prefix}" -name "${name}" -print -quit 2>/dev/null || true
  fi
}

cmake_prefixes=()
BREW_PREFIX="$("${BREW_CMD[@]}" --prefix)"
arpack_prefix="$("${BREW_CMD[@]}" --prefix arpack)"

if [[ -n "${arpack_prefix}" ]]; then
  arpack_cmake_file="$(find_cmake_dir "${arpack_prefix}" "*ARPACK*Config.cmake")"
  if [[ -n "${arpack_cmake_file}" ]]; then
    cmake_prefixes+=("$(dirname "${arpack_cmake_file}")")
  fi
fi

cmake_prefixes+=("${BREW_PREFIX}")

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p "${BUILD_DIR}"
fi

arch_flags="x86_64"
cmake_prefix_path="$(IFS=';'; echo "${cmake_prefixes[*]}")"

log "Configuring CMake build in ${BUILD_DIR}..."
cmake_args=(
  -S "${REPO_ROOT}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DVEGAFEM_USE_MKL=OFF
  -DVEGAFEM_BUILD_COPYLEFT=OFF
  -DCMAKE_OSX_ARCHITECTURES="${arch_flags}"
  -DCMAKE_PREFIX_PATH="${cmake_prefix_path}"
)

cmake "${cmake_args[@]}"

log "Building VegaFEM..."
cmake --build "${BUILD_DIR}" --config Release -- -j"$(sysctl -n hw.logicalcpu)"

example_bin="${BUILD_DIR}/utilities/${TARGET_EXE_NAME}"
if [[ ! -x "${example_bin}" ]]; then
  log "Build succeeded but ${example_bin} was not found."
  exit 1
fi

RUN_AFTER_BUILD="${RUN_AFTER_BUILD:-0}"
if [[ "${RUN_AFTER_BUILD}" == "1" ]]; then
  log "Running ${TARGET_EXE_NAME} with ${TARGET_CONFIG}..."
  (cd "${REPO_ROOT}/examples/turtle" && "${example_bin}" "${TARGET_CONFIG}")
else
  log "Skipping auto-run (set RUN_AFTER_BUILD=1 to enable)."
fi
