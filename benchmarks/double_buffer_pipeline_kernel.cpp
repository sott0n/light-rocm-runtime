#include <stdint.h>

extern "C" __global__ void
double_buffer_pipeline_kernel(uint32_t *data, uint32_t count, uint32_t rounds) {
  constexpr uint32_t block_size = 256;
  const uint32_t index = __builtin_amdgcn_workgroup_id_x() * block_size +
                         __builtin_amdgcn_workitem_id_x();
  if (index >= count) {
    return;
  }

  uint32_t value = data[index];
  for (uint32_t round = 0; round < rounds; ++round) {
    value = value * 1664525u + 1013904223u;
  }
  data[index] = value;
}
