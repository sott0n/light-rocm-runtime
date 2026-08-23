extern "C" __global__ void
mutex_contention_wait_kernel(unsigned long long iterations,
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

extern "C" __global__ void mutex_contention_noop_kernel(int token) {
  if (threadIdx.x == 0 && token == -1) {
    asm volatile("");
  }
}
