extern "C" __global__ void launch_benchmark_kernel(int token) {
  if (token < 0) {
    __builtin_trap();
  }
}
