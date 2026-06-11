#ifndef LRRT_EXAMPLES_COMMON_EXAMPLE_UTILS_H_
#define LRRT_EXAMPLES_COMMON_EXAMPLE_UTILS_H_

#include <stdio.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace lrrt_example {

inline std::vector<unsigned char> read_file(const char *path) {
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

} // namespace lrrt_example

#endif // LRRT_EXAMPLES_COMMON_EXAMPLE_UTILS_H_
