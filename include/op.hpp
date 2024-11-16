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
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
      sum += static_cast<float>(x[i]) * static_cast<float>(x[i]);
    }
    sum /= size;
    float epsilon = 1e-5f;
    sum += epsilon;
    sum = 1.0f / sqrtf(sum);
    // normalize and scale
    for (size_t i = 0; i < size; i++) {
      o[i] = static_cast<T>(static_cast<float>(weight[i]) * (sum * x[i]));
    }
  }
};

template <typename T>
class MatMul {
 public:
  static Tensor<T> Compute(const Tensor<T>& left, const Tensor<T>& right) {
    CHECK_EQ(left.GetShape()[1], right.GetShape()[0])
        << "Inner dimensions should be the same";
    CHECK_EQ(left.GetShape().GetRank(), 2)
        << "Input tensor should be 2D tensor";
    CHECK_EQ(right.GetShape()[1], 1) << "Input tensor should be vector";
    Tensor<T> out(Shape({left.GetShape()[0]}));

    // left : (n,d), right : (d,1) -> out : (n,1)
    const size_t n = left.GetShape()[0];
    const size_t d = left.GetShape()[1];
    Compute(out.GetData(), left.GetData(), right.GetData(), n, d);
    return out;
  }
  static void Compute(T* out, const T* left, const T* right, const size_t n,
                      const size_t d) {
    CHECK_GE(n, 0) << "Size 'n' should be greater than or equal to 0";
    CHECK_GE(d, 0) << "Size 'd' should be greater than or equal to 0";
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    for (size_t i = 0; i < d; i++) {
      T val = static_cast<T>(0.0f);
      for (size_t j = 0; j < n; j++) {
        val += right[i * n + j] * left[j];
      }
      out[i] = val;
    }
  }
};
}  // namespace llama2
