extern "C" __global__ void
async_copy_launch_benchmark_kernel(const float *input, float *output,
                                   float alpha, int index) {
  if (threadIdx.x == 0) {
    output[0] = alpha * input[index];
  }
}
