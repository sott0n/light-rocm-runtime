#include "light_rocr/transport/kfd/code_cache.hpp"
#include "light_rocr/transport/kfd/executable_image.hpp"
#include "light_rocr/transport/kfd/session.hpp"

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

namespace {

struct TestContext {
  int failures = 0;

  void expect(bool condition, const std::string &message) {
    if (!condition) {
      ++failures;
      std::cerr << "  FAIL: " << message << '\n';
    }
  }
};

using TestFunction = std::function<void(TestContext *)>;

void rejects_unowned_resources_without_allocation(TestContext *context) {
  light_rocr::transport::kfd::KfdSession session;
  light_rocr::runtime::Node node;
  light_rocr::transport::kfd::AqlQueue queue;
  light_rocr::transport::kfd::ExecutableImage image;
  const auto frozen = light_rocr::transport::kfd::freeze_executable_image(
      session, node, queue, image, std::chrono::steady_clock::now());
  context->expect(
      frozen.status.error ==
              light_rocr::transport::kfd::CodeCacheError::InvalidArgument &&
          !frozen.operation,
      "invalid cache resources created a pending operation");
}

void empty_operation_release_is_idempotent(TestContext *context) {
  light_rocr::transport::kfd::CodeCacheOperation operation;
  light_rocr::transport::kfd::AqlQueue queue;
  context->expect(!operation, "default cache operation owns resources");
  context->expect(static_cast<bool>(operation.release(queue)),
                  "default cache operation release failed");
  context->expect(static_cast<bool>(operation.release(queue)),
                  "repeated cache operation release failed");
}

void enum_names(TestContext *context) {
  context->expect(
      std::string(light_rocr::transport::kfd::code_cache_error_name(
          light_rocr::transport::kfd::CodeCacheError::DestroyQueue)) ==
          "destroy_queue",
      "unexpected code-cache error name");
}

static_assert(!std::is_copy_constructible_v<
              light_rocr::transport::kfd::CodeCacheOperation>);
static_assert(
    !std::is_copy_assignable_v<light_rocr::transport::kfd::CodeCacheOperation>);
static_assert(std::is_nothrow_move_constructible_v<
              light_rocr::transport::kfd::CodeCacheOperation>);

} // namespace

int main() {
  const std::pair<const char *, TestFunction> tests[] = {
      {"rejects_unowned_resources_without_allocation",
       rejects_unowned_resources_without_allocation},
      {"empty_operation_release_is_idempotent",
       empty_operation_release_is_idempotent},
      {"enum_names", enum_names},
  };

  int failures = 0;
  for (const auto &[name, test] : tests) {
    std::cout << "[ RUN      ] " << name << '\n';
    TestContext context;
    test(&context);
    failures += context.failures;
    std::cout << (context.failures == 0 ? "[       OK ] " : "[  FAILED  ] ")
              << name << '\n';
  }
  return failures == 0 ? 0 : 1;
}
