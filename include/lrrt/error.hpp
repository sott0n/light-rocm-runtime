#ifndef LRRT_ERROR_HPP_
#define LRRT_ERROR_HPP_

#include "lrrt/lrrt.h"

#include <stdexcept>
#include <string>

namespace lrrt {

class Error : public std::runtime_error {
public:
  Error(lr_status_t status, const char *operation)
      : std::runtime_error(std::string(operation) +
                           " failed: " + lr_status_string(status)),
        status_(status) {}

  lr_status_t status() const { return status_; }

private:
  lr_status_t status_;
};

inline void check(lr_status_t status, const char *operation) {
  if (status != LR_SUCCESS) {
    throw Error(status, operation);
  }
}

} // namespace lrrt

#endif // LRRT_ERROR_HPP_
