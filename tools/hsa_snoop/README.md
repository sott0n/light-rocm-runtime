# hsa-snoop tools

These scripts build
[`hsa-snoop`](https://github.com/sbates130272/hsa-snoop) and run an LRRT
program under its AQL and SDMA queue tracer.

## Setup

Fetch and build the pinned hsa-snoop version:

```sh
tools/hsa_snoop/setup.sh
```

The source checkout and build output are stored under the ignored
`build-hsa-snoop/` directory. The resulting executable is:

```text
build-hsa-snoop/build/hsa-snoop
```

To include the optional Prometheus exporter:

```sh
tools/hsa_snoop/setup.sh --prometheus
```

Run `tools/hsa_snoop/setup.sh --help` for source directory, build directory,
revision, and parallel-job options.

## Capture an LRRT trace

Build an LRRT executable, then pass its command after `--`:

```sh
tools/hsa_snoop/run.sh -- \
  ./build-bench/lrrt_async_copy_launch_benchmark 100
```

The wrapper invokes `sudo` because hsa-snoop requires root access to tracefs,
process memory, and pagemap. The default Perfetto trace is written to:

```text
build-hsa-snoop/lrrt.pftrace
```

Open the trace with [Perfetto UI](https://ui.perfetto.dev/).

Common options:

```sh
tools/hsa_snoop/run.sh \
  --debug \
  --poll-us 5 \
  --format json \
  --out build-hsa-snoop/lrrt.json \
  -- ./build-bench/lrrt_launch_overhead_benchmark 1000
```

- `--debug` enables hsa-snoop decoder diagnostics.
- `--poll-us N` changes the queue polling interval.
- `--format perfetto|json` selects the trace format.
- `--out PATH` selects the output file.
- `--duration SEC` limits the capture duration.
- `--dry-run` prints the generated command without executing it.
- `--no-sudo` disables automatic sudo invocation.

Run `tools/hsa_snoop/run.sh --help` for the complete option list.

Native Linux capture requires an accessible `/dev/kfd`. hsa-snoop v1.0.0
calibrates SDMA decoding for CDNA SDMA v4.x; validate SDMA packet decoding
before relying on copy counts from other GPU families such as gfx1101.
