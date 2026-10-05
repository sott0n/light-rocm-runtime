# light-rocr Performance Compared with ROCr

This document records a development comparison between the ROCr backend and
the direct KFD `light-rocr` backend. It is a diagnostic snapshot, not a
hardware performance guarantee.

## Summary

The direct KFD backend now outperforms ROCr in the measured launch and Qwen
paths:

| Area | Result | Verdict |
| --- | --- | --- |
| Qwen correctness | Identical top-five logits | Equivalent |
| Qwen end-to-end | Direct KFD 43.9-44.5 ms; ROCr 51.2 ms | Direct KFD is 13-14% faster in both backend orders |
| Qwen CPU submission | Direct KFD 8.70-8.73 ms; ROCr 10.43-10.56 ms | Direct KFD is about 17% faster |
| Empty-kernel sustained throughput | Direct KFD 72.6k/s; ROCr 66.0k/s | Direct KFD is about 10% faster |
| Synchronized launch round trip | Direct KFD 30.9 us; ROCr 47.4 us | Direct KFD is about 35% faster |

The earlier approximately 2.5x Qwen slowdown came from model pages remaining
in system memory after CPU writes to host-mapped VRAM. GPU-only VRAM
initialization removed that gap. Placing executable images in VRAM rather than
host-visible GTT subsequently removed the remaining empty-kernel throughput
deficit. The Qwen comparison now also favors direct KFD, although fixed-clock
measurements are still required for a hardware-independent performance claim.

## Measurement conditions

| Item | Value |
| --- | --- |
| Runtime revision | `f312a8a` plus the direct KFD timestamp change |
| Date | 2026-10-05 |
| GPU | AMD Radeon RX 7800 XT (`gfx1101`, 60 CUs) |
| CPU | AMD Ryzen Threadripper 3970X, 32 cores / 64 threads |
| ROCm | 6.4.4-129 |
| Build | `Release`, identical source and Triton kernel bundles |
| ROCr path | LRRT -> `libhsa-runtime64` -> KFD |
| light-rocr path | LRRT -> direct KFD transport -> KFD |

The two backends were measured in alternating order. The system was otherwise
idle. GPU performance level remained `auto`; setting a fixed performance level
was not available without administrator privileges. Qwen used 350 in-process
warm-up iterations, and SCLK was sampled throughout each process.

## Launch overhead

The table reports one 10,000-launch check after the executable placement
change.

| Metric | ROCr | Direct KFD | Direct KFD relative to ROCr |
| --- | ---: | ---: | ---: |
| Host enqueue, final sync excluded | 8.137 us | 0.492 us | 94.0% lower |
| Submit and synchronize, one final sync | 15.158 us | 13.776 us | 9.1% lower |
| Launch round trip, sync after every launch | 47.360 us | 30.932 us | 34.7% lower |
| Sustained throughput | 65,970 launches/s | 72,591 launches/s | 10.0% higher |

The direct KFD producer has a much cheaper host enqueue path. Executable images
now use GPU-local VRAM, matching ROCr's placement policy and avoiding
instruction fetches through host-visible GTT. With that change, direct KFD also
wins the measured batch throughput and synchronized round trip.

## Qwen workload

The model measurement uses the converted FP32 Qwen2.5-0.5B-Instruct bundle:

| Property | Value |
| --- | ---: |
| Decoder layers | 24 |
| Valid keys | 3 |
| Dispatches per iteration | 5,618 |
| Live device allocation | 1,887.943 MiB |
| Model tail | Final RMSNorm and 151,936-element logits |

Each process performed 350 warm-up iterations after model setup and then 50
recorded iterations. The second pair reversed backend order.

| Pair | ROCr | Direct KFD | Observation |
| --- | ---: | ---: | --- |
| ROCr then direct KFD | 51.150 ms | 43.895 ms | Direct KFD was 14.2% faster |
| Direct KFD then ROCr | 51.226 ms | 44.509 ms | Direct KFD was 13.1% faster |

Automatic DPM still raised SCLK during the recorded interval. In the first
pair, ROCr entered the interval at approximately 1.85 GHz and reached 2.08 GHz;
direct KFD entered at approximately 1.74 GHz and reached 2.15 GHz. The second
pair followed the same trajectory. Direct KFD therefore did not win by being
measured at a higher initial clock, and reversing backend order did not change
the result. Fixed-clock repetition remains desirable, but the former
order-dependent ambiguity is no longer present in these measurements.

CPU submission time is stable because it does not depend on GPU clock. Summing
the benchmark's six decoder-stage submission measurements over 24 layers and
three keys gives:

| Backend | CPU submission per stack |
| --- | ---: |
| ROCr | 10.43-10.56 ms |
| Direct KFD | 8.70-8.73 ms |

The CPU submission gap is about 1.8 ms per stack, but submission overlaps GPU
execution and therefore cannot be subtracted directly from the end-to-end gap.
Direct KFD uses persistent kernarg arenas, reusable
completion signals, direct queue-index atomics, and a direct MMIO doorbell
store. The ROCr path goes through the HSA entry points and its queue, lifetime,
signal, and kernarg bookkeeping.

Direct KFD event profiling now separates the device interval from the CPU
round trip. A Release diagnostic pass after 350 warm-up iterations measured:

| Backend | CPU round trip | Profiled GPU interval | Decoder-stage CPU submission |
| --- | ---: | ---: | ---: |
| ROCr | 48.981 ms | 41.684 ms | approximately 10.55 ms |
| Direct KFD | 44.101 ms | 44.383 ms | approximately 8.90 ms |

The start marker is synchronized before the stack is submitted. Leaving that
marker pending materially perturbs the ROCr path and produces a non-comparable
interval. The CPU and GPU columns are separate passes, so their small run-to-run
difference must not be interpreted as an exact subtraction.

This result rules out faster kernel execution as the source of the direct KFD
end-to-end win: ROCr's measured device interval was about 6% shorter. The
advantage is on the host runtime path. Direct KFD submits the stack about 16%
faster and returns from the complete submit-and-synchronize round trip sooner;
the residual is consistent with additional completion-retirement and queue
synchronization work in the ROCr path. A phase profile inside ROCr queue
synchronization is still needed to divide that remaining host cost exactly.

The following diagnostic A/B changes did not explain the remaining gap:

- disabling ROCr queue profiling changed Qwen from 51.150 ms to 51.035 ms;
- using mailbox-free `AMD_GPU_ONLY` signals for ROCr internal dispatch
  retirement changed empty-kernel throughput from 65,970/s to 67,388/s;
- changing the direct KFD AQL ring from dedicated `AQL_QUEUE_MEM` GTT to the
  same USERPTR backing used by ROCr changed throughput from 72,591/s to
  72,793/s.

The two paths use identical kernel text and AQL packet construction. Their KFD
queues also use the same ring size, queue percentage, priority, EOP size, and
CWSR sizes. Executable allocations use the same
`VRAM | WRITABLE | EXECUTABLE | NO_SUBSTITUTE` policy.

Consequently, the Qwen advantage is attributable to the lighter host runtime
path rather than faster kernels. Fixed-clock repetition is still required for
a hardware-independent percentage, but it does not change which side of the
runtime/device boundary contains the measured advantage.

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
  -DLRRT_ENABLE_TRITON_EXAMPLES=ON \
  -DLRRT_BUILD_TRITON_BENCHMARKS=ON \
  -DLRRT_AMDGPU_TARGET=gfx1101

cmake -S . -B build-kfd-perf \
  -DCMAKE_BUILD_TYPE=Release \
  -DLRRT_BACKEND=light-rocr \
  -DLRRT_LIGHT_ROCR_TRANSPORT=kfd \
  -DLRRT_BUILD_BENCHMARKS=ON \
  -DLRRT_ENABLE_TRITON_EXAMPLES=ON \
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
  --weights-dir /path/to/qwen-bundle --layers 24 --valid-keys 3 \
  --warmup-iterations 350

./build-kfd-perf/lrrt_triton_mini_decoder_layer_benchmark 50 \
  --weights-dir /path/to/qwen-bundle --layers 24 --valid-keys 3 \
  --warmup-iterations 350
```

For an end-to-end latency claim, fix the GPU performance level or record SCLK
throughout each run. Otherwise, report the observed range and backend order.
