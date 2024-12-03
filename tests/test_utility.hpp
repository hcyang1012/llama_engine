#pragma once

#include <cmath>
#include <cstddef>

namespace llama::test {
bool tensor_float_compare(const float* a, const float* b, const size_t size,
                          const float epsilon) {
  bool equal = true;
  for (size_t i = 0; i < size; i++) {
    if (std::abs(a[i] - b[i]) > epsilon) {
      equal = false;
      break;
    }
  }
  return equal;
};
}  // namespace llama::test