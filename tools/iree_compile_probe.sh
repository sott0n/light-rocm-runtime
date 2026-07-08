#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/iree_compile_probe.sh [options]

Compiles a tiny MLIR program through IREE's ROCm/HIP path far enough to inspect
HAL executable metadata for the lrrt IREE adapter investigation.

Options:
  --iree-compile PATH  iree-compile executable
                       (default: build-iree-tools/tools/iree-compile)
  --input PATH         MLIR input (default: tools/iree_minimal_mul.mlir)
  --lld-dir DIR        Directory containing lld for ROCm HSACO linking
                       (default: /opt/rocm/llvm/bin when present)
  --out-dir DIR        Output directory (default: build-iree-probe)
  --target CHIP        ROCm target chip (default: gfx1101)
  --try-vmfb           Also attempt full VMFB serialization. This may fail when
                       the available ld.lld is incompatible with IREE's LLVM.
  -h, --help           Show this help
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
iree_compile="${repo_root}/build-iree-tools/tools/iree-compile"
input="${repo_root}/tools/iree_minimal_mul.mlir"
lld_dir=""
out_dir="${repo_root}/build-iree-probe"
target="gfx1101"
try_vmfb=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  --iree-compile)
    iree_compile="$2"
    shift 2
    ;;
  --input)
    input="$2"
    shift 2
    ;;
  --lld-dir)
    lld_dir="$2"
    shift 2
    ;;
  --out-dir)
    out_dir="$2"
    shift 2
    ;;
  --target)
    target="$2"
    shift 2
    ;;
  --try-vmfb)
    try_vmfb=1
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

if [[ ! -x "${iree_compile}" ]]; then
  cat >&2 <<EOF
iree-compile was not found or is not executable:
  ${iree_compile}

Build the local IREE tools first:
  CMAKE_GENERATOR=Ninja tools/build_iree_tools.sh --jobs 16
EOF
  exit 1
fi

if [[ ! -f "${input}" ]]; then
  echo "input MLIR file not found: ${input}" >&2
  exit 1
fi

if [[ -z "${lld_dir}" && -x /opt/rocm/llvm/bin/lld ]]; then
  lld_dir="/opt/rocm/llvm/bin"
fi

if [[ -n "${lld_dir}" ]]; then
  if [[ ! -x "${lld_dir}/lld" ]]; then
    echo "lld was not found or is not executable: ${lld_dir}/lld" >&2
    exit 1
  fi
  export PATH="${lld_dir}:${PATH}"
fi

mkdir -p "${out_dir}"

base="${out_dir}/minimal_mul_${target}"
config_ir="${base}_executable_configurations.mlir"
target_ir="${base}_executable_targets.mlir"
vmfb="${base}.vmfb"
vmfb_log="${base}_vmfb.log"

compile_common=(
  "${iree_compile}"
  "--iree-hal-target-device=hip"
  "--iree-rocm-target=${target}"
)

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run "${compile_common[@]}" \
  --compile-to=executable-configurations \
  "${input}" \
  -o "${config_ir}"

run "${compile_common[@]}" \
  --compile-to=executable-targets \
  "${input}" \
  -o "${target_ir}"

vmfb_status="skipped"
if [[ "${try_vmfb}" -eq 1 ]]; then
  if "${compile_common[@]}" "${input}" -o "${vmfb}" >"${vmfb_log}" 2>&1; then
    vmfb_status="ok"
  else
    vmfb_status="failed"
  fi
fi

echo
echo "IREE ROCm compile probe"
echo "  input: ${input}"
echo "  target: ${target}"
echo "  lld: $(command -v lld || true)"
echo "  executable-configurations: ${config_ir}"
echo "  executable-targets: ${target_ir}"
echo "  vmfb: ${vmfb_status}"
if [[ "${vmfb_status}" == "failed" ]]; then
  echo "  vmfb-log: ${vmfb_log}"
fi

echo
echo "Metadata anchors from executable-targets:"
grep -n -E \
  'hal\.executable|hal\.executable\.variant|hal\.executable\.export|pipeline\.layout|workgroup_size|subgroup_size|llvm\.func|rocdl\.kernel|rocdl\.reqd_work_group_size|stream\.cmd\.dispatch' \
  "${target_ir}" || true
