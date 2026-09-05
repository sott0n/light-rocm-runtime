extern "C" __global__ void launch_benchmark_kernel(int token) {
  if (token < 0) {
    __builtin_trap();
  }
}

extern "C" __global__ void launch_benchmark_delay(unsigned long long iterations,
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
