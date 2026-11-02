# Plan

## Goal

Provide a small AMDGPU execution runtime for compiler-generated HSACO without
requiring HIP. Keep ROCr as the reference backend and provide a self-authored
`gfx1101` path whose userspace boundary is Linux KFD.

## Current acceptance gate

The direct KFD backend must implement the same LRRT C API and observable
lifetime, ordering, timing, and error behavior used by the ROCr backend. It must
run the repository's Clang, Triton, and IREE workloads without dynamically
linking ROCr, `libhsakmt`, or HIP. Full 24-layer Qwen prompt/decode correctness
and the shared backend test suite are the end-to-end gates.

## Scope

- Linux KFD and one `gfx1101` GPU
- AMDHSA ET_DYN code objects produced by the repository's compiler paths
- device and pinned/pageable host memory
- multiple AQL queues, events, dependencies, scratch, and GPU timing
- LRRT's C ABI rather than HSA or HIP API compatibility

Peer access, managed memory, graph capture, stream priorities, full multi-GPU
scheduling, math libraries, and a general HSA-compatible shared library remain
outside this project scope.
