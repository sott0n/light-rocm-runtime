extern "C" __global__ void fill(float *out, float value, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = value;
  }
}
