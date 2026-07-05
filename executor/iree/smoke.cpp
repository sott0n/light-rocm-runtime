#include "iree_adapter.hpp"

int main() {
  try {
    lrrt::executor::iree::require_unsupported("smoke");
  } catch (const lrrt::executor::iree::UnsupportedFeature &) {
    return 0;
  }
  return 1;
}
