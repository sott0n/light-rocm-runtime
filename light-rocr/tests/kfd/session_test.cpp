#include "light_rocr/transport/kfd/session.hpp"

#include <cerrno>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

static_assert(sizeof(light_rocr::transport::kfd::VmResult) > 0,
              "session.hpp must provide the acquire_vm result type");
static_assert(sizeof(light_rocr::transport::kfd::UserSignalResult) > 0,
              "session.hpp must provide the create_user_signal result type");

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

void version_contract(TestContext *context) {
  using light_rocr::runtime::KfdVersion;
  using light_rocr::transport::kfd::is_supported_kfd_version;

  context->expect(is_supported_kfd_version(KfdVersion{1, 1}),
                  "minimum KFD version was rejected");
  context->expect(is_supported_kfd_version(KfdVersion{1, 16}),
                  "newer compatible KFD minor version was rejected");
  context->expect(!is_supported_kfd_version(KfdVersion{1, 0}),
                  "older KFD version was accepted");
  context->expect(!is_supported_kfd_version(KfdVersion{2, 0}),
                  "unknown KFD major version was accepted");
  context->expect(std::string(light_rocr::transport::kfd::session_error_name(
                      light_rocr::transport::kfd::SessionError::InspectKfd)) ==
                      "inspect_kfd",
                  "new session error has no stable name");
}

void missing_device_is_reported(TestContext *context) {
  const auto opened = light_rocr::transport::kfd::KfdSession::open(
      "/definitely/missing/light-rocr-kfd");
  context->expect(!opened, "missing KFD device unexpectedly opened");
  context->expect(opened.status.error ==
                      light_rocr::transport::kfd::SessionError::OpenKfd,
                  "wrong missing-device error");
  context->expect(opened.status.system_error == ENOENT,
                  "missing-device errno was not preserved");
}

void ioctl_failure_is_reported(TestContext *context) {
  const auto opened = light_rocr::transport::kfd::KfdSession::open("/dev/null");
  context->expect(!opened, "/dev/null unexpectedly behaved as KFD");
  context->expect(opened.status.error ==
                      light_rocr::transport::kfd::SessionError::QueryKfdVersion,
                  "wrong ioctl failure error");
  context->expect(opened.status.system_error != 0,
                  "ioctl failure errno was not preserved");
}

void invalid_session_rejects_signal_creation(TestContext *context) {
  light_rocr::transport::kfd::KfdSession session;
  const light_rocr::runtime::Node node;
  auto created = session.create_user_signal(node, 1);
  context->expect(!created, "invalid session unexpectedly created a signal");
  context->expect(
      created.status.error ==
          light_rocr::transport::kfd::UserSignalError::InvalidSession,
      "wrong invalid-session signal error");
  context->expect(!created.signal,
                  "invalid session returned signal resource ownership");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, TestFunction>> tests = {
      {"version contract", version_contract},
      {"missing device", missing_device_is_reported},
      {"ioctl failure", ioctl_failure_is_reported},
      {"invalid signal session", invalid_session_rejects_signal_creation},
  };

  int failures = 0;
  for (const auto &[name, test] : tests) {
    TestContext context;
    test(&context);
    if (context.failures == 0) {
      std::cout << "PASS: " << name << '\n';
    } else {
      std::cerr << "FAIL: " << name << " (" << context.failures << ")\n";
      failures += context.failures;
    }
  }
  return failures == 0 ? 0 : 1;
}
