extern "C" __global__ void lrrt_copy_bytes(const unsigned char *source,
                                           unsigned char *destination,
                                           unsigned long size,
                                           unsigned long stride,
                                           unsigned long workgroup_size) {
  const unsigned long index =
      static_cast<unsigned long>(blockIdx.x) * workgroup_size + threadIdx.x;
  for (unsigned long offset = index; offset < size; offset += stride) {
    destination[offset] = source[offset];
  }
}
