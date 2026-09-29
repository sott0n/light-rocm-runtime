extern "C" __global__ void vector_add(const float *a, const float *b, float *c,
                                      int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) {
    return;
  }

  // A dynamically indexed private array forces the code object to reserve a
  // non-zero private segment. Keep the calculation observable so the dispatch
  // test verifies that the queue's scratch backing is actually usable.
  volatile float scratch[17];
  const int slot = (threadIdx.x * 5 + 3) % 17;
  scratch[slot] = a[index];
  c[index] = scratch[slot] + b[index];
}
