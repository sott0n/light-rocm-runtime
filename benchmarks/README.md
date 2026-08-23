# Benchmarks

This directory contains microbenchmarks for launch, synchronization, transfer,
and host/device pipeline overhead. The Qwen Triton model benchmark is described
separately in the [Qwen execution guide](../examples/qwen/README.md#triton).

## Build and run

From the repository root:

```sh
cmake -S . -B build-bench -DLRRT_BUILD_BENCHMARKS=ON
cmake --build build-bench -j2

./build-bench/lrrt_launch_overhead_benchmark
./build-bench/lrrt_mutex_contention_benchmark
./build-bench/lrrt_async_copy_launch_benchmark
./build-bench/lrrt_pinned_host_transfer_benchmark
./build-bench/lrrt_double_buffer_pipeline_benchmark
```

| Executable | Default workload |
| --- | --- |
| `lrrt_launch_overhead_benchmark` | 10,000 sustained launches, a 512-launch burst, and 1,000 synchronized round trips |
| `lrrt_mutex_contention_benchmark` | Lightweight API probes during queue, event, backpressure, and synchronous-copy waits |
| `lrrt_async_copy_launch_benchmark` | 100 iterations of copy/launch and launch/copy dependency paths |
| `lrrt_pinned_host_transfer_benchmark` | 4 KiB through 1 GiB H2D and D2H transfers; adaptive iteration count targeting 256 MiB per row |
| `lrrt_double_buffer_pipeline_benchmark` | 50 x 4 MiB chunks with 64 GPU compute rounds per chunk |

Run benchmarks on an otherwise idle system and repeat them when comparing
changes. GPU clock management, CPU scheduling, system load, and ROCm revisions
can materially change these numbers.

## Development result snapshot

The following results are a diagnostic baseline, not a performance guarantee.
They are from one run on 2026-08-10 using the default arguments and a working
tree based on revision `a8dd546`.

| Component | Value |
| --- | --- |
| GPU | AMD Radeon RX 7800 XT (`gfx1101`, 60 CUs) |
| CPU | AMD Ryzen Threadripper 3970X, 32 cores / 64 threads |
| ROCm | 6.4.4-129 |
| HSA runtime | 1.15, extension 1.7 |
| Kernel | Linux 6.8.0-124-generic |

### Launch overhead

| Metric | Time |
| --- | ---: |
| Idle synchronize, no queued work | 0.629 us |
| Host enqueue, 512-launch burst | 8.699 us/launch |
| Device batch interval | 14.349 us/launch |
| Submit and synchronize, one final sync | 14.743 us/launch |
| Launch round trip, one sync per launch | 47.780 us/launch |
| Sustained throughput | 67,827 launches/s |

The host-enqueue measurement excludes the final synchronization. The device
batch interval is measured with HSA event timestamps.

### Asynchronous dependencies

The copy-to-launch cases use a 4 MiB D2D copy. The launch-to-copy case uses a
4-byte D2D copy after the kernel.

| Dependency chain | Path | Submission | End-to-end | Submission speedup | End-to-end speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Copy -> launch | Explicit host wait | 64.665 us | 120.110 us | - | - |
| Copy -> launch | Device dependency | 2.101 us | 122.604 us | 30.78x | 0.98x |
| Copy -> default marker | Explicit host wait | 64.117 us | 93.638 us | - | - |
| Copy -> default marker | Device dependency | 1.928 us | 94.789 us | 33.26x | 0.99x |
| Launch -> copy | Explicit host wait | 54.948 us | 76.581 us | - | - |
| Launch -> copy | Device dependency | 1.821 us | 53.305 us | 30.17x | 1.44x |

Device-side dependencies primarily remove host submission stalls. They do not
necessarily reduce completed end-to-end time when the same device work remains
on the critical path.

### Host transfer

This condensed table shows representative rows from the full 4 KiB-to-1 GiB
sweep. `Call` is host API time, `Wait` is event wait time, and bandwidth uses
the steady-state `Total` time.

| Direction | Memory | Size | Call | Wait | Total | Bandwidth |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| H2D | Pageable, async | 4 KiB | 3.58 us | 21.15 us | 24.73 us | 0.15 GiB/s |
| H2D | Pinned, async | 4 KiB | 1.42 us | 23.26 us | 24.68 us | 0.15 GiB/s |
| D2H | Pageable, async | 4 KiB | 3.66 us | 20.79 us | 24.45 us | 0.16 GiB/s |
| D2H | Pinned, async | 4 KiB | 1.49 us | 21.74 us | 23.23 us | 0.16 GiB/s |
| H2D | Pageable, async | 1 MiB | 3.61 us | 78.43 us | 82.04 us | 11.90 GiB/s |
| H2D | Pinned, async | 1 MiB | 1.33 us | 81.26 us | 82.59 us | 11.82 GiB/s |
| D2H | Pageable, async | 1 MiB | 3.59 us | 90.53 us | 94.11 us | 10.38 GiB/s |
| D2H | Pinned, async | 1 MiB | 1.38 us | 93.09 us | 94.47 us | 10.34 GiB/s |
| H2D | Pageable, async | 64 MiB | 3.88 us | 3,549.76 us | 3,553.64 us | 17.59 GiB/s |
| H2D | Pinned, async | 64 MiB | 1.42 us | 3,545.33 us | 3,546.75 us | 17.62 GiB/s |
| D2H | Pageable, async | 64 MiB | 3.94 us | 4,371.16 us | 4,375.11 us | 14.29 GiB/s |
| D2H | Pinned, async | 64 MiB | 1.40 us | 4,371.30 us | 4,372.70 us | 14.29 GiB/s |
| H2D | Pageable, async | 1 GiB | 12.19 us | 55,931.13 us | 55,943.33 us | 17.88 GiB/s |
| H2D | Pinned, async | 1 GiB | 2.46 us | 55,078.20 us | 55,080.66 us | 18.16 GiB/s |
| D2H | Pageable, async | 1 GiB | 21.42 us | 69,346.82 us | 69,368.24 us | 14.42 GiB/s |
| D2H | Pinned, async | 1 GiB | 2.61 us | 68,985.96 us | 68,988.57 us | 14.50 GiB/s |

### Double-buffer pipeline

| Mode | Total | CPU prepare | Throughput | Input bandwidth |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 682.841 ms | 38.194 ms | 73.2 chunks/s | 0.286 GiB/s |
| Double-buffered | 634.236 ms | 38.312 ms | 78.8 chunks/s | 0.308 GiB/s |

The double-buffered path saved 48.605 ms (7.1%) and produced a 1.077x
end-to-end speedup. Total time includes CPU preparation, H2D transfer, GPU work,
and the final drain; allocation and validation are excluded.

## Interpreting and reproducing results

- Pass a positive iteration count to the launch and dependency benchmarks.
- Pass a positive GPU wait iteration count to the mutex contention benchmark.
- Use `--max-size-mib`, `--iterations`, and `--warmup` to control the host
  transfer sweep.
- Use `--chunks`, `--warmup-chunks`, `--chunk-size-mib`, and
  `--compute-rounds` to control the pipeline workload.
- Set `NO_COLOR=1` when capturing output for automated processing.
- Compare multiple interleaved runs from the same build and machine rather than
  treating this snapshot as a hardware specification.
