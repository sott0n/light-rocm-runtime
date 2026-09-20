# light-rocr

`light-rocr` is an experimental AMDGPU runtime and AMDHSA code-object loader
for `light-rocm-runtime`. Its goal is to replace ROCr and its loader while
keeping Linux KFD as the kernel boundary.

It does not use `libhsa-runtime64.so`. GPU access currently goes through
`libhsakmt`. The direct KFD transport currently covers device discovery, VM
acquisition, and GPU-mapped host-visible GTT allocation; queue execution
remains on the `libhsakmt` path.

## Boundary

LRRT constructs and publishes AQL packets and owns pending dispatch resources.
`light-rocr` provides queue indices, Ring memory, doorbells, and KFD-facing
object lifetimes; it does not decide what a kernel-dispatch packet contains.

## Status

The current target is `gfx1101`. `light-rocr` has run the full 24-layer
Qwen2.5-0.5B IREE path, including prompt prefill, per-layer device-resident KV
caches, the model tail, and autoregressive generation. Its generated tokens and
top logits matched the ROCr backend. Clang and Triton generated HSACOs are also
covered by the GPU tests.

## Build

```sh
cmake -S light-rocr -B build-light-rocr
cmake --build build-light-rocr
ctest --test-dir build-light-rocr --output-on-failure
```

The GPU tools are built when the `libhsakmt`, DRM, and NUMA development files
are available. For a host-only build:

```sh
cmake -S light-rocr -B build-light-rocr \
  -DLIGHT_ROCR_ENABLE_HSAKMT=OFF
```

To include `light-rocr` in the main project build, configure with
`-DLRRT_BACKEND=light-rocr`. See the
[Qwen execution guide](../examples/qwen/README.md) for the IREE model build and
execution flow.

## Try it

Inspect an HSACO without a GPU:

```sh
./build-light-rocr/light-rocr-inspect-hsaco PATH_TO_HSACO
```

On a `gfx1101` machine with access to `/dev/kfd`, enable and run the GPU E2E
test against the repository's normal Clang vector-add HSACO:

```sh
cmake -S light-rocr -B build-light-rocr-gpu \
  -DLIGHT_ROCR_BUILD_GPU_TESTS=ON \
  -DLIGHT_ROCR_VECTOR_ADD_HSACO="$PWD/build/vector_add_kernel.hsaco"
cmake --build build-light-rocr-gpu
ctest --test-dir build-light-rocr-gpu --output-on-failure
```
