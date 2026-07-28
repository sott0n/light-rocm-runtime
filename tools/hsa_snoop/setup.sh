#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/hsa_snoop/setup.sh [options]

Fetch and build the hsa-snoop version used by the LRRT tracing workflow.

Options:
  --ref REF          Git tag or branch (default: v1.0.0)
  --source-dir DIR   Checkout directory
                     (default: build-hsa-snoop/src)
  --build-dir DIR    Build directory
                     (default: build-hsa-snoop/build)
  --prometheus       Build the optional Prometheus exporter
  --jobs N           Parallel build jobs (default: number of online CPUs)
  -h, --help         Show this help
EOF
}

lrrt_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
hsa_snoop_ref="v1.0.0"
hsa_snoop_source="${lrrt_root}/build-hsa-snoop/src"
hsa_snoop_build="${lrrt_root}/build-hsa-snoop/build"
hsa_snoop_prometheus="OFF"
hsa_snoop_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

while [[ $# -gt 0 ]]; do
  case "$1" in
  --ref)
    hsa_snoop_ref="$2"
    shift 2
    ;;
  --source-dir)
    hsa_snoop_source="$2"
    shift 2
    ;;
  --build-dir)
    hsa_snoop_build="$2"
    shift 2
    ;;
  --prometheus)
    hsa_snoop_prometheus="ON"
    shift
    ;;
  --jobs)
    hsa_snoop_jobs="$2"
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

if [[ ! "${hsa_snoop_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--jobs must be a positive integer" >&2
  exit 2
fi

if [[ -e "${hsa_snoop_source}" && ! -d "${hsa_snoop_source}/.git" ]]; then
  echo "source path exists but is not a Git checkout: ${hsa_snoop_source}" >&2
  exit 1
fi

if [[ ! -d "${hsa_snoop_source}/.git" ]]; then
  mkdir -p "$(dirname "${hsa_snoop_source}")"
  git clone --branch "${hsa_snoop_ref}" --depth 1 \
    https://github.com/sbates130272/hsa-snoop.git "${hsa_snoop_source}"
else
  current_commit="$(git -C "${hsa_snoop_source}" rev-parse HEAD)"
  expected_commit="$(git -C "${hsa_snoop_source}" rev-parse \
    "${hsa_snoop_ref}^{commit}" 2>/dev/null || true)"
  if [[ -z "${expected_commit}" || "${current_commit}" != "${expected_commit}" ]]; then
    current_ref="$(git -C "${hsa_snoop_source}" describe --tags --exact-match \
      2>/dev/null || echo "${current_commit}")"
    cat >&2 <<EOF
hsa-snoop checkout is at ${current_ref}, not ${hsa_snoop_ref}:
  ${hsa_snoop_source}

Use matching --ref/--source-dir values, or remove this generated checkout and
run the setup again.
EOF
    exit 1
  fi
fi

cmake -S "${hsa_snoop_source}" -B "${hsa_snoop_build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DHSA_SNOOP_PROMETHEUS="${hsa_snoop_prometheus}"
cmake --build "${hsa_snoop_build}" --parallel "${hsa_snoop_jobs}"

echo
echo "hsa-snoop is ready:"
echo "  ${hsa_snoop_build}/hsa-snoop"
