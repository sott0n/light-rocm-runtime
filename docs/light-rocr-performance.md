# light-rocr Performance Compared with ROCr

This document records a development comparison between the ROCr backend and
the direct KFD `light-rocr` backend. It is a diagnostic snapshot, not a
hardware performance guarantee.

## Summary

The direct KFD backend is now performance-competitive with ROCr, but it has not
yet demonstrated equal or better end-to-end performance. The conservative
result is:

| Area | Result | Verdict |
| --- | --- | --- |
| Qwen correctness | Identical top-five logits | Equivalent |
| Qwen end-to-end | Direct KFD 58.9-59.8 ms; ROCr 47.5-61.4 ms | Inconclusive under automatic DPM; direct KFD is 24% slower than the best observed ROCr result |
| Qwen CPU submission | Direct KFD 9.9 ms; ROCr 14.1 ms | Direct KFD is about 30% faster |
| Empty-kernel sustained throughput | Direct KFD 57.5k/s; ROCr 64.3k/s | Direct KFD is about 11% slower |
| Synchronized launch round trip | Direct KFD 35.1 us; ROCr 48.8 us | Direct KFD is about 28% faster |

This is a large improvement over the earlier approximately 2.5x Qwen
slowdown. That gap was caused by model pages remaining in system memory after
CPU writes to host-mapped VRAM, and GPU-only VRAM initialization removed it.
What remains is not a general 2.5x runtime penalty: direct KFD wins the measured
host-side paths, loses empty-kernel batch throughput, and still needs a
fixed-clock Qwen comparison before end-to-end parity can be claimed.

## Measurement conditions

| Item | Value |
| --- | --- |
| Revision | `bbd4ddb` |
| Date | 2026-10-04 |
| GPU | AMD Radeon RX 7800 XT (`gfx1101`, 60 CUs) |
| CPU | AMD Ryzen Threadripper 3970X, 32 cores / 64 threads |
| ROCm | 6.4.4-129 |
| Build | `Release`, identical source and Triton kernel bundles |
| ROCr path | LRRT -> `libhsa-runtime64` -> KFD |
| light-rocr path | LRRT -> direct KFD transport -> KFD |

The two backends were measured in alternating order. The system was otherwise
idle. GPU performance level remained `auto`; setting a fixed performance level
was not available without administrator privileges.

## Launch overhead

The table reports the median of three alternating runs of 10,000 empty-kernel
launches.

| Metric | ROCr | Direct KFD | Direct KFD relative to ROCr |
| --- | ---: | ---: | ---: |
| Host enqueue, final sync excluded | 8.337 us | 0.492 us | 94.1% lower |
| Submit and synchronize, one final sync | 15.543 us | 17.380 us | 11.8% higher |
| Launch round trip, sync after every launch | 48.814 us | 35.053 us | 28.2% lower |
| Sustained throughput | 64,337 launches/s | 57,538 launches/s | 10.6% lower |

The direct KFD producer has a much cheaper host enqueue path. Its lower
sustained throughput indicates that queue consumption and synchronization,
rather than AQL packet construction, account for the remaining empty-kernel
batch difference.

## Qwen workload

The model measurement uses the converted FP32 Qwen2.5-0.5B-Instruct bundle:

| Property | Value |
| --- | ---: |
| Decoder layers | 24 |
| Valid keys | 3 |
| Dispatches per iteration | 5,618 |
| Live device allocation | 1,887.943 MiB |
| Model tail | Final RMSNorm and 151,936-element logits |

Each recorded process followed a separate 100-iteration warm-up process for
the same backend. The recorded process then ran 50 iterations, including its
own five warm-up iterations. The second pair reversed backend order.

| Pair | ROCr | Direct KFD | Observation |
| --- | ---: | ---: | --- |
| ROCr then direct KFD | 61.410 ms | 59.766 ms | Direct KFD was 2.7% faster in this order |
| Direct KFD then ROCr | 47.511 ms | 58.914 ms | Direct KFD was 24.0% slower in this order |

These values must not be averaged into a backend speedup claim. During longer
runs, the observed GPU SCLK rose from approximately 454 MHz to 2,680 MHz. The
same workload also fell from approximately 150 ms to approximately 60 ms as
the clock increased. Process-level warm-up did not make the automatic DPM
state reproducible across backend order.

Therefore, the current evidence supports "competitive and no longer
pathologically slower," but not "faster than ROCr." The best observed ROCr
result remains lower than the direct KFD results.

CPU submission time is stable because it does not depend on GPU clock. Summing
the benchmark's six decoder-stage submission measurements over 24 layers and
three keys gives:

| Backend | CPU submission per stack |
| --- | ---: |
| ROCr | 14.09-14.11 ms |
| Direct KFD | 9.89-9.93 ms |

The direct KFD result includes constant-time reuse of dispatch owners. Before
that change, removing a reusable owner from the front of a 5,618-element
vector caused quadratic pointer movement; decoder-stage submission was about
43% higher.

## Correctness

Both backends produced finite outputs and the same top five logits:

| Rank | Token | Logit |
| ---: | ---: | ---: |
| 1 | 117367 | 14.702360 |
| 2 | 95144 | 14.483202 |
| 3 | 141759 | 14.318941 |
| 4 | 28796 | 13.477837 |
| 5 | 41337 | 13.465656 |

The direct KFD binary does not dynamically depend on ROCr, `libhsakmt`, or the
HIP runtime.

## Reproducing the comparison

Create separate Release builds so both backends use the same compiler options:

```sh
cmake -S . -B build-rocr-perf \
  -DCMAKE_BUILD_TYPE=Release \
  -DLRRT_BACKEND=rocr \
  -DLRRT_BUILD_BENCHMARKS=ON \
  -DLRRT_BUILD_TRITON_BENCHMARKS=ON \
  -DLRRT_AMDGPU_TARGET=gfx1101

cmake -S . -B build-kfd-perf \
  -DCMAKE_BUILD_TYPE=Release \
  -DLRRT_BACKEND=light-rocr \
  -DLRRT_LIGHT_ROCR_TRANSPORT=kfd \
  -DLRRT_BUILD_BENCHMARKS=ON \
  -DLRRT_BUILD_TRITON_BENCHMARKS=ON \
  -DLRRT_AMDGPU_TARGET=gfx1101

cmake --build build-rocr-perf -j2
cmake --build build-kfd-perf -j2
```

Run launch overhead with the same iteration count:

```sh
./build-rocr-perf/lrrt_launch_overhead_benchmark 10000
./build-kfd-perf/lrrt_launch_overhead_benchmark 10000
```

After preparing a Qwen bundle as described in the
[Qwen execution guide](../examples/qwen/README.md#triton), run both backends
with identical arguments:

```sh
./build-rocr-perf/lrrt_triton_mini_decoder_layer_benchmark 50 \
  --weights-dir /path/to/qwen-bundle --layers 24 --valid-keys 3

./build-kfd-perf/lrrt_triton_mini_decoder_layer_benchmark 50 \
  --weights-dir /path/to/qwen-bundle --layers 24 --valid-keys 3
```

For an end-to-end latency claim, fix the GPU performance level or record SCLK
throughout each run. Otherwise, report the observed range and backend order.
