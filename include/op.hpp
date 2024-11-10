/**
 * @file op.hpp
 * @brief Header for the Various Operations.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-11
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <glog/logging.h>

#include <vector>
// Project Headers

// Third-party Headers

namespace llama2 {
template <typename T>
class RmsNorm {
 public:
  static std::vector<T> Compute(const std::vector<T>& x,
                                const std::vector<T>& weight) {
    CHECK_EQ(x.size(), weight.size())
        << "Size of x and weight should be the same" << x.size() << " vs "
        << weight.size();
    std::vector<T> o(x.size());
    Compute(o.data(), x.data(), weight.data(), x.size());
    return o;
  }

  static void Compute(T* o, const T* x, const T* weight, const size_t size) {
    CHECK_GE(size, 0) << "Size should be greater than or equal to 0";
    // calculate sum of squares
    T sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
      sum += x[i] * x[i];
    }
    sum /= size;
    const T epsilon = static_cast<T>(1e-5f);
    sum += epsilon;
    sum = 1.0f / sqrtf(sum);
    // normalize and scale
    for (size_t i = 0; i < size; i++) {
      o[i] = weight[i] * (sum * x[i]);
    }
  }
};
}  // namespace llama2
