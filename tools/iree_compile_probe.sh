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
  --iree-run-module PATH
                       iree-run-module executable
                       (default: build-iree-tools/tools/iree-run-module)
  --input PATH         MLIR input (default: tools/iree_minimal_mul.mlir)
  --lld-dir DIR        Directory containing lld for ROCm HSACO linking
                       (default: /opt/rocm/llvm/bin when present)
  --summary PATH       Write executable-target metadata JSON summary
  --out-dir DIR        Output directory (default: build-iree-probe)
  --target CHIP        ROCm target chip (default: gfx1101)
  --emit-hsaco         Emit a raw HSACO image from executable-targets MLIR for
                       lrrt module loading experiments.
  --hsaco PATH         Raw HSACO output path for --emit-hsaco
                       (default: <out-dir>/minimal_mul_<target>.hsaco)
  --try-vmfb           Also attempt full VMFB serialization. This may fail when
                       the available ld.lld is incompatible with IREE's LLVM.
  --run-baseline       Run the serialized VMFB with iree-run-module --device=hip
                       and validate the expected minimal_mul output. This
                       implies --try-vmfb.
  --baseline-function NAME
                       Function to run for --run-baseline (default: simple_mul)
  --baseline-input VALUE
                       Add an iree-run-module --input value for --run-baseline.
                       May be repeated. Defaults match tools/iree_minimal_mul.mlir.
  --baseline-expected VALUE
                       Expected output line for --run-baseline
                       (default: 4xf32=10 40 90 160)
  -h, --help           Show this help
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
iree_compile="${repo_root}/build-iree-tools/tools/iree-compile"
iree_run_module="${repo_root}/build-iree-tools/tools/iree-run-module"
input="${repo_root}/tools/iree_minimal_mul.mlir"
lld_dir=""
out_dir="${repo_root}/build-iree-probe"
summary_path=""
target="gfx1101"
emit_hsaco=0
try_vmfb=0
run_baseline=0
baseline_function="simple_mul"
baseline_inputs=()
baseline_expected="4xf32=10 40 90 160"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --iree-compile)
    iree_compile="$2"
    shift 2
    ;;
  --iree-run-module)
    iree_run_module="$2"
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
  --summary)
    summary_path="$2"
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
  --emit-hsaco)
    emit_hsaco=1
    shift
    ;;
  --hsaco)
    hsaco_path="$2"
    shift 2
    ;;
  --try-vmfb)
    try_vmfb=1
    shift
    ;;
  --run-baseline)
    run_baseline=1
    try_vmfb=1
    shift
    ;;
  --baseline-function)
    baseline_function="$2"
    shift 2
    ;;
  --baseline-input)
    baseline_inputs+=("$2")
    shift 2
    ;;
  --baseline-expected)
    baseline_expected="$2"
    shift 2
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

if [[ "${run_baseline}" -eq 1 && ! -x "${iree_run_module}" ]]; then
  cat >&2 <<EOF
iree-run-module was not found or is not executable:
  ${iree_run_module}

Build the local IREE tools first:
  CMAKE_GENERATOR=Ninja tools/build_iree_tools.sh --jobs 16
EOF
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
default_summary="${base}_metadata.json"
vmfb="${base}.vmfb"
vmfb_log="${base}_vmfb.log"
baseline_log="${base}_baseline.log"
default_hsaco="${base}.hsaco"

if [[ -z "${summary_path}" ]]; then
  summary_path="${default_summary}"
fi
if [[ -z "${hsaco_path:-}" ]]; then
  hsaco_path="${default_hsaco}"
fi

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

run "${repo_root}/tools/iree_metadata_summary.py" \
  "${target_ir}" \
  -o "${summary_path}"

hsaco_status="skipped"
if [[ "${emit_hsaco}" -eq 1 ]]; then
  run "${iree_compile}" \
    --compile-mode=hal-executable \
    "--iree-rocm-target=${target}" \
    --iree-rocm-container-type=hsaco \
    "${target_ir}" \
    -o "${hsaco_path}"
  hsaco_status="ok"
fi

vmfb_status="skipped"
if [[ "${try_vmfb}" -eq 1 ]]; then
  if "${compile_common[@]}" "${input}" -o "${vmfb}" >"${vmfb_log}" 2>&1; then
    vmfb_status="ok"
  else
    vmfb_status="failed"
  fi
fi

baseline_status="skipped"
baseline_failure=""
if [[ "${run_baseline}" -eq 1 ]]; then
  if [[ "${vmfb_status}" != "ok" ]]; then
    baseline_status="blocked"
    baseline_failure="VMFB serialization did not complete"
  else
    if [[ "${#baseline_inputs[@]}" -eq 0 ]]; then
      baseline_inputs=(
        "4xf32=1 2 3 4"
        "4xf32=10 20 30 40"
      )
    fi

    baseline_cmd=(
      "${iree_run_module}"
      "--device=hip"
      "--module=${vmfb}"
      "--function=${baseline_function}"
    )
    for baseline_input in "${baseline_inputs[@]}"; do
      baseline_cmd+=("--input=${baseline_input}")
    done
    baseline_cmd+=("--output=-")

    if "${baseline_cmd[@]}" >"${baseline_log}" 2>&1; then
      if grep -Fxq "${baseline_expected}" "${baseline_log}"; then
        baseline_status="ok"
      else
        baseline_status="mismatch"
        baseline_failure="expected output line was not found"
      fi
    else
      baseline_status="failed"
      baseline_failure="iree-run-module exited with an error"
    fi
  fi
fi

echo
echo "IREE ROCm compile probe"
echo "  input: ${input}"
echo "  target: ${target}"
echo "  lld: $(command -v lld || true)"
echo "  executable-configurations: ${config_ir}"
echo "  executable-targets: ${target_ir}"
echo "  metadata-summary: ${summary_path}"
echo "  hsaco: ${hsaco_status}"
if [[ "${emit_hsaco}" -eq 1 ]]; then
  echo "  hsaco-output: ${hsaco_path}"
fi
echo "  vmfb: ${vmfb_status}"
if [[ "${vmfb_status}" == "failed" ]]; then
  echo "  vmfb-log: ${vmfb_log}"
fi
echo "  baseline: ${baseline_status}"
if [[ "${run_baseline}" -eq 1 ]]; then
  echo "  baseline-runner: ${iree_run_module}"
  echo "  baseline-function: ${baseline_function}"
  echo "  baseline-expected: ${baseline_expected}"
  echo "  baseline-log: ${baseline_log}"
  if [[ -n "${baseline_failure}" ]]; then
    echo "  baseline-failure: ${baseline_failure}"
  fi
fi

echo
echo "Metadata anchors from executable-targets:"
grep -n -E \
  'hal\.executable|hal\.executable\.variant|hal\.executable\.export|pipeline\.layout|workgroup_size|subgroup_size|llvm\.func|rocdl\.kernel|rocdl\.reqd_work_group_size|stream\.cmd\.dispatch' \
  "${target_ir}" || true

if [[ "${run_baseline}" -eq 1 && "${baseline_status}" != "ok" ]]; then
  exit 1
fi
