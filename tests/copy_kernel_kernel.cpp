extern "C" __global__ void copy_kernel(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef LRRT_TEST_HIDDEN_KERNARGS
  if (gridDim.x != 3 || gridDim.y != 1 || gridDim.z != 1 || blockDim.x != 64 ||
      blockDim.y != 1 || blockDim.z != 1) {
    if (i == 0) {
      out[0] = -1.0f;
    }
    return;
  }
#endif
  if (i < n) {
    out[i] = in[i];
  }
}
