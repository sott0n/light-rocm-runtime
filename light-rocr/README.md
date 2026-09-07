# light-rocr

`light-rocr` is an experimental AMDGPU runtime and AMDHSA code-object loader
for `light-rocm-runtime`. Its goal is to replace ROCr and its loader while
keeping Linux KFD as the kernel boundary.

It does not use `libhsa-runtime64.so`. GPU access currently goes through
`libhsakmt`; a direct KFD transport will replace that dependency later.

## Status

The current target is `gfx1101`. A built-in kernel can run end to end through a
self-authored AQL queue and signal. Normal HSACO files can be parsed and mapped
into GPU memory, but executing a loaded HSACO end to end is not yet supported.

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
`-DLRRT_ENABLE_LIGHT_ROCR=ON`.

## Try it

Inspect an HSACO without a GPU:

```sh
./build-light-rocr/light-rocr-inspect-hsaco PATH_TO_HSACO
```

On a `gfx1101` machine with access to `/dev/kfd`, materialize a normal HSACO in
GPU-visible memory:

```sh
./build-light-rocr/light-rocr-check-image PATH_TO_HSACO
```

Run the built-in kernel through the current AQL dispatch path:

```sh
./build-light-rocr/light-rocr-check-dispatch
```
