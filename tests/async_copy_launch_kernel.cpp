extern "C" __global__ void async_copy_launch_kernel(const float *in, float *out,
                                                    float alpha, int index) {
  if (threadIdx.x == 0) {
    out[0] = alpha * in[index];
  }
}

extern "C" __global__ void queue_wait_kernel(unsigned long long iterations,
                                             unsigned long long *output) {
  if (threadIdx.x != 0) {
    return;
  }
  volatile unsigned long long value = iterations;
  for (unsigned long long i = 0; i < iterations; ++i) {
    value = value * 2862933555777941757ULL + 3037000493ULL;
  }
  output[0] = value;
}
