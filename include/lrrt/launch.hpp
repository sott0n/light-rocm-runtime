#ifndef LRRT_LAUNCH_HPP_
#define LRRT_LAUNCH_HPP_

#include "lrrt/module.hpp"

#include <vector>

namespace lrrt {

inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size) {
  check(lr_launch(kernel, &config, args, args_size), "lr_launch");
}

inline void launch(const Queue &queue, lr_kernel_t *kernel,
                   const lr_launch_config_t &config, const void *args,
                   size_t args_size,
                   const std::vector<const Event *> &dependencies = {}) {
  if (dependencies.empty()) {
    check(lr_launch_on_queue(queue.get(), kernel, &config, args, args_size),
          "lr_launch_on_queue");
    return;
  }
  std::vector<lr_event_t *> handles = event_handles(dependencies);
  check(lr_launch_on_queue_with_dependencies(queue.get(), kernel, &config, args,
                                             args_size, handles.data(),
                                             handles.size()),
        "lr_launch_on_queue_with_dependencies");
}

template <typename Args>
inline void launch(const Queue &queue, lr_kernel_t *kernel,
                   const lr_launch_config_t &config, const Args &args,
                   const std::vector<const Event *> &dependencies = {}) {
  launch(queue, kernel, config, &args, sizeof(args), dependencies);
}

inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size,
                   const std::vector<const Event *> &dependencies) {
  std::vector<lr_event_t *> handles = event_handles(dependencies);
  check(lr_launch_with_dependencies(kernel, &config, args, args_size,
                                    handles.data(), handles.size()),
        "lr_launch_with_dependencies");
}

template <typename Args>
inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const Args &args,
                   const std::vector<const Event *> &dependencies) {
  launch(kernel, config, &args, sizeof(args), dependencies);
}

template <typename Args>
inline void launch(lr_kernel_t *kernel, const lr_launch_config_t &config,
                   const Args &args) {
  launch(kernel, config, &args, sizeof(args));
}

inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size) {
  launch(kernel.get(), config, args, args_size);
}

inline void launch(const Queue &queue, const Kernel &kernel,
                   const lr_launch_config_t &config, const void *args,
                   size_t args_size,
                   const std::vector<const Event *> &dependencies = {}) {
  launch(queue, kernel.get(), config, args, args_size, dependencies);
}

inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const void *args, size_t args_size,
                   const std::vector<const Event *> &dependencies) {
  launch(kernel.get(), config, args, args_size, dependencies);
}

template <typename Args>
inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const Args &args) {
  launch(kernel.get(), config, args);
}

template <typename Args>
inline void launch(const Queue &queue, const Kernel &kernel,
                   const lr_launch_config_t &config, const Args &args,
                   const std::vector<const Event *> &dependencies = {}) {
  launch(queue, kernel.get(), config, args, dependencies);
}

template <typename Args>
inline void launch(const Kernel &kernel, const lr_launch_config_t &config,
                   const Args &args,
                   const std::vector<const Event *> &dependencies) {
  launch(kernel.get(), config, args, dependencies);
}

} // namespace lrrt

#endif // LRRT_LAUNCH_HPP_
