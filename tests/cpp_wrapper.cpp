#include "lrrt/lrrt.hpp"

#include <math.h>
#include <stdio.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef LRRT_SCALE_HSACO
#define LRRT_SCALE_HSACO "scale_kernel.hsaco"
#endif

typedef struct scale_args_t {
  const float *in;
  float *out;
  float alpha;
  int n;
} scale_args_t;

static std::vector<unsigned char> read_file(const char *path) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    throw std::runtime_error(std::string("failed to open ") + path);
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fclose(file);
    throw std::runtime_error(std::string("failed to seek ") + path);
  }
  long length = ftell(file);
  if (length <= 0) {
    fclose(file);
    throw std::runtime_error(std::string("empty file ") + path);
  }
  rewind(file);

  std::vector<unsigned char> data(static_cast<size_t>(length));
  if (fread(data.data(), 1, data.size(), file) != data.size()) {
    fclose(file);
    throw std::runtime_error(std::string("failed to read ") + path);
  }
  fclose(file);
  return data;
}

static void expect_lrrt_error(lr_status_t status) {
  try {
    lrrt::check(status, "expected_failure");
  } catch (const lrrt::Error &error) {
    if (error.status() == status) {
      return;
    }
    throw std::runtime_error("lrrt::Error carried the wrong status");
  }
  throw std::runtime_error("lrrt::check did not throw");
}

int main(void) {
  try {
    expect_lrrt_error(LR_ERROR_INVALID_ARGUMENT);

    lrrt::Runtime runtime;

    uint32_t count = runtime.device_count();
    if (count == 0) {
      printf("cpp_wrapper: skipped, no GPU devices\n");
      return 0;
    }

    lrrt::Device device = runtime.open_device(0);
    if (device.index() != 0) {
      throw std::runtime_error("Device reported the wrong index");
    }

    const int n = 64;
    const float alpha = 3.0f;
    float in[n];
    float out[n];
    for (int i = 0; i < n; ++i) {
      in[i] = (float)i + 1.0f;
      out[i] = 0.0f;
    }

    lrrt::DeviceBuffer first_in(device, sizeof(in));
    lrrt::DeviceBuffer device_in(std::move(first_in));
    lrrt::DeviceBuffer device_out(device, sizeof(out) / 2);
    device_out = lrrt::DeviceBuffer(device, sizeof(out));
    if (device_in.size() != sizeof(in) || device_out.size() != sizeof(out)) {
      throw std::runtime_error("DeviceBuffer reported the wrong size");
    }

    lrrt::copy_to_device(device_in, in);

    lrrt::DeviceBuffer device_copy(device, sizeof(in));
    lrrt::copy_device_to_device(device_copy, device_in, sizeof(in));
    float copied[n];
    lrrt::copy_to_host(copied, device_copy);
    for (int i = 0; i < n; ++i) {
      if (fabsf(copied[i] - in[i]) > 0.001f) {
        throw std::runtime_error("device copy result mismatch");
      }
    }

    std::vector<float> vector_in(in, in + n);
    std::vector<float> vector_out(n, 0.0f);
    lrrt::copy_to_device(device_in, vector_in);
    lrrt::copy_to_host(vector_out, device_in);
    for (int i = 0; i < n; ++i) {
      if (fabsf(vector_out[i] - in[i]) > 0.001f) {
        throw std::runtime_error("vector copy result mismatch");
      }
    }

    std::vector<unsigned char> hsaco = read_file(LRRT_SCALE_HSACO);
    lrrt::Module first_module(device, hsaco);
    lrrt::Module module(std::move(first_module));
    module = lrrt::Module(device, hsaco);
    if (module.get() == nullptr) {
      throw std::runtime_error("Module returned a null handle");
    }
    lrrt::Kernel kernel = module.kernel("scale");
    if (kernel.get() == nullptr) {
      throw std::runtime_error("Kernel returned a null handle");
    }

    scale_args_t args = {
        (const float *)device_in.data(),
        (float *)device_out.data(),
        alpha,
        n,
    };
    lr_launch_config_t config = {{64, 1, 1}, {64, 1, 1}, 0};
    lrrt::launch(kernel, config, args);
    device.synchronize();

    lrrt::copy_to_host(out, device_out);

    for (int i = 0; i < n; ++i) {
      float expected = alpha * in[i];
      if (fabsf(out[i] - expected) > 0.001f) {
        throw std::runtime_error("scale result mismatch");
      }
    }

    printf("cpp_wrapper: ok\n");
    return 0;
  } catch (const std::exception &error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
