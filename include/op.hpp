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

#include "tensor.hpp"

// Third-party Headers

namespace llama2 {
template <typename T>
class RmsNorm {
 public:
  static Tensor<T> Compute(const Tensor<T>& x, const Tensor<T>& weight) {
    CHECK(x.GetShape() == weight.GetShape())
        << "Size of the input tensors should be the same";
    CHECK_EQ(x.GetShape().GetRank(), 1) << "Input tensor should be 1D tensor";
    Tensor<T> o(x.GetShape());
    Compute(o.GetData(), x.GetData(), weight.GetData(), x.GetShape()[0]);
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
