extern "C" __global__ void async_copy_launch_kernel(const float *in, float *out,
                                                    float alpha, int index) {
  if (threadIdx.x == 0) {
    out[0] = alpha * in[index];
  }
}
