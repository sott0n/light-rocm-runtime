#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/hsa_snoop/run.sh [options] -- <command> [args...]

Run an LRRT program under hsa-snoop and write a Perfetto or JSON trace. The
wrapper uses sudo when it is not already running as root.

Options:
  --hsa-snoop PATH  hsa-snoop executable
                    (default: build-hsa-snoop/build/hsa-snoop, then PATH)
  --out PATH        Trace output (default: build-hsa-snoop/lrrt.pftrace)
  --format FORMAT   perfetto (default) or json
  --poll-us N       Ring polling interval (default: hsa-snoop's 20 us)
  --duration SEC    Stop tracing after SEC seconds
  --debug           Set HSA_SNOOP_DEBUG=1
  --no-sudo         Do not invoke sudo; useful when already granted capabilities
  --dry-run         Print the command without running it
  -h, --help        Show this help

Example:
  tools/hsa_snoop/run.sh -- \
    ./build-bench/lrrt_async_copy_launch_benchmark 100
EOF
}

print_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

lrrt_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
default_hsa_snoop="${lrrt_root}/build-hsa-snoop/build/hsa-snoop"
hsa_snoop_bin=""
trace_format="perfetto"
trace_out="${lrrt_root}/build-hsa-snoop/lrrt.pftrace"
trace_out_set=0
poll_us=""
duration=""
debug_enabled=0
use_sudo=1
dry_run=0
ran_with_sudo=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  --hsa-snoop)
    hsa_snoop_bin="$2"
    shift 2
    ;;
  --out)
    trace_out="$2"
    trace_out_set=1
    shift 2
    ;;
  --format)
    trace_format="$2"
    shift 2
    ;;
  --poll-us)
    poll_us="$2"
    shift 2
    ;;
  --duration)
    duration="$2"
    shift 2
    ;;
  --debug)
    debug_enabled=1
    shift
    ;;
  --no-sudo)
    use_sudo=0
    shift
    ;;
  --dry-run)
    dry_run=1
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "a command is required after --" >&2
  usage >&2
  exit 2
fi

if [[ "${trace_format}" != "perfetto" && "${trace_format}" != "json" ]]; then
  echo "--format must be perfetto or json" >&2
  exit 2
fi

if [[ "${trace_format}" == "json" && "${trace_out_set}" -eq 0 ]]; then
  trace_out="${lrrt_root}/build-hsa-snoop/lrrt.json"
fi

if [[ -n "${poll_us}" && ! "${poll_us}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--poll-us must be a positive integer" >&2
  exit 2
fi

if [[ -n "${duration}" && ! "${duration}" =~ ^[0-9]+$ ]]; then
  echo "--duration must be a non-negative integer" >&2
  exit 2
fi

if [[ -z "${hsa_snoop_bin}" ]]; then
  if [[ -x "${default_hsa_snoop}" ]]; then
    hsa_snoop_bin="${default_hsa_snoop}"
  elif command -v hsa-snoop >/dev/null 2>&1; then
    hsa_snoop_bin="$(command -v hsa-snoop)"
  else
    cat >&2 <<EOF
hsa-snoop was not found. Build the pinned version first:
  tools/hsa_snoop/setup.sh
EOF
    exit 1
  fi
fi

if [[ ! -x "${hsa_snoop_bin}" ]]; then
  echo "hsa-snoop is not executable: ${hsa_snoop_bin}" >&2
  exit 1
fi

if [[ "${dry_run}" -eq 0 && ! -e /dev/kfd ]] &&
  ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  cat >&2 <<'EOF'
/dev/kfd is unavailable. Native hsa-snoop tracing requires an AMDGPU/KFD
device. Run this on the host with the GPU device exposed; WSL2 is handled by
hsa-snoop's librocdxg mode.
EOF
  exit 1
fi

if [[ "${trace_out}" != /* ]]; then
  trace_out="${PWD}/${trace_out}"
fi
mkdir -p "$(dirname "${trace_out}")"

snoop_args=(
  "${hsa_snoop_bin}"
  --format "${trace_format}"
  --out "${trace_out}"
)
if [[ -n "${poll_us}" ]]; then
  snoop_args+=(--poll-us "${poll_us}")
fi
if [[ -n "${duration}" ]]; then
  snoop_args+=(--duration "${duration}")
fi
snoop_args+=(-- "$@")

run_args=()
if [[ "${debug_enabled}" -eq 1 ]]; then
  run_args+=(env HSA_SNOOP_DEBUG=1)
fi
if [[ "${use_sudo}" -eq 1 && "${EUID}" -ne 0 ]]; then
  run_args=(sudo "${run_args[@]}")
  ran_with_sudo=1
fi
run_args+=("${snoop_args[@]}")

print_command "${run_args[@]}"
if [[ "${dry_run}" -eq 1 ]]; then
  exit 0
fi

"${run_args[@]}"

if [[ -f "${trace_out}" && "${ran_with_sudo}" -eq 1 ]]; then
  sudo chown "$(id -u):$(id -g)" "${trace_out}"
elif [[ -f "${trace_out}" && "${EUID}" -eq 0 && -n "${SUDO_UID:-}" ]]; then
  chown "${SUDO_UID}:${SUDO_GID:-${SUDO_UID}}" "${trace_out}"
fi

echo
echo "Trace written to:"
echo "  ${trace_out}"
echo "Open it with https://ui.perfetto.dev/"
