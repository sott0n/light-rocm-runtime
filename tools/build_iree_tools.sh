#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/build_iree_tools.sh [options]

Builds the IREE tools needed by lrrt IREE adapter validation in a separate
build directory. This script does not run as part of the lrrt CMake build.

Options:
  --source DIR       IREE source tree (default: third_party/iree)
  --build-dir DIR    IREE tool build directory (default: build-iree-tools)
  --target CHIP      ROCm test target chip. Empty disables IREE ROCm tests
                     (default: empty)
  --jobs N           Parallel build jobs (default: 2)
  --dry-run          Print commands without executing them
  -h, --help         Show this help
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/third_party/iree"
build_dir="${repo_root}/build-iree-tools"
target_chip=""
jobs="2"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  --source)
    source_dir="$2"
    shift 2
    ;;
  --build-dir)
    build_dir="$2"
    shift 2
    ;;
  --target)
    target_chip="$2"
    shift 2
    ;;
  --jobs)
    jobs="$2"
    shift 2
    ;;
  --dry-run)
    dry_run=1
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${dry_run}" -eq 0 ]]; then
    "$@"
  fi
}

if [[ ! -f "${source_dir}/runtime/src/iree/base/api.h" ]]; then
  cat >&2 <<EOF
IREE source tree not found at:
  ${source_dir}

Initialize the submodule first:
  git submodule update --init third_party/iree

Or pass --source /path/to/iree.
EOF
  exit 1
fi

run cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF \
  -DIREE_BUILD_BENCHMARKS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_ROCM=ON \
  -DIREE_HAL_DRIVER_HIP=ON \
  -DIREE_ROCM_TEST_TARGET_CHIP="${target_chip}"

run cmake --build "${build_dir}" --target iree-compile iree-run-module \
  -j"${jobs}"
