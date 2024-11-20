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

#include <cmath>
#include <vector>
// Project Headers

#include "config.hpp"
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

class RoPE {
 public:
  static void Compute(const Tensor<float>& input, const size_t position,
                      const Config& config, Tensor<float>& output,
                      const bool overwrite = false) {
    CHECK_EQ(input.GetShape()[0], config.Dim())
        << "Input tensor should have the same dimension as the config";

    if (overwrite) {
      CHECK_EQ(output.GetShape()[0], config.Dim())
          << "Output tensor should have the same dimension as the config";
    } else {
      output = Tensor<float>(input.GetShape());
    }

    compute(input.GetData(), position, config, output.GetData());
  }

 private:
  static void compute(const float* input, const size_t position,
                      const Config& config, float* output) {
    const size_t dim = config.Dim();
    const size_t kv_dim =
        (config.Dim() * config.NumKVHeads()) / config.NumHeads();
    const size_t head_size = dim / config.NumHeads();
    for (size_t i = 0; i < dim; i += 2) {
      size_t head_dim = i % head_size;
      float theta =
          1.0f / powf(10000.0f, head_dim / static_cast<float>(head_size));
      float val = position * theta;
      float fcr = cosf(val);
      float fci = sinf(val);
      size_t rotn =
          i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (size_t v = 0; v < rotn; v++) {
        float v0 = input[i];
        float v1 = input[i + 1];
        output[i] = v0 * fcr - v1 * fci;
        output[i + 1] = v0 * fci + v1 * fcr;
      }
    }
  }
};

template <typename T>
class SoftMax {
 public:
  static void Compute(const Tensor<T>& input, Tensor<float>& output) {
    CHECK_EQ(input.GetShape(), output.GetShape())
        << "Input and output tensor should have the same shape";
    CHECK_LE(input.GetShape().GetRank(), 2)
        << "Input tensor should be 1D or 2D tensor";

    const size_t kStride = input.GetShape().GetRank() == 1
                               ? input.GetShape()[0]
                               : input.GetShape()[1];

    if (input.GetShape().GetRank() == 1) {
      Compute(input.GetData(), output.GetData(), kStride);
    } else {
      const size_t kBatchSize = input.GetShape()[0];
      for (size_t batch = 0; batch < kBatchSize; batch++) {
        Compute(input.GetData() + batch * kStride,
                output.GetData() + batch * kStride, kStride);
      }
    }
  }

  static void Compute(const T* input, float* output, const size_t size) {
    CHECK_GE(size, 0) << "Size should be greater than or equal to 0";
    float sum = 0.0f;
    float max_val = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < size; i++) {
      max_val = std::max(max_val, static_cast<float>(input[i]));
    }
    for (size_t i = 0; i < size; i++) {
      sum += expf(static_cast<float>(input[i]) - max_val);
    }
    for (size_t i = 0; i < size; i++) {
      output[i] = expf(static_cast<float>(input[i]) - max_val) / sum;
    }
  }

 private:
};

template <typename T>
class Attention {
 public:
  /**
   * @brief
   *
   * @param Q  (kPerHeadDim)
   * @param K (kHeadDim, Seq_Len)
   * @param V (kHeadDim, Seq_Len)
   * @param config
   * @param pos
   * @param header_idx
   * @param output {kPerHeadDim}
   */
  static void Compute(const Tensor<T>& Q, const Tensor<T>& K,
                      const Tensor<T>& V, const Config& config,
                      const size_t pos, const size_t header_idx,
                      Tensor<T>& output) {
    const size_t kPerHeadDim = config.Dim() / config.NumHeads();
    const size_t kKVHeadDim =
        (config.Dim() * config.NumKVHeads()) / config.NumHeads();
    CHECK_EQ(Q.GetShape()[0], kPerHeadDim);
    CHECK_EQ(K.GetShape()[0], kKVHeadDim);
    CHECK_EQ(V.GetShape()[0], kKVHeadDim);

    Tensor<T> attention_scores({pos + 1});
    for (size_t t = 0; t <= pos; ++t) {
      float score = 0.0f;
      for (size_t i = 0; i < kPerHeadDim; ++i) {
        score += (Q[{i}] * K[{i, t}]);
      }
      score /= sqrtf(kPerHeadDim);
      attention_scores[{t}] = score;
    }

    // Calculate the attention score and store it back to the same buffer
    SoftMax<T>::Compute(attention_scores, attention_scores);

    // Weighted sum of the values, store back into output
    std::fill(output.GetData(), output.GetData() + output.GetShape().GetSize(),
              static_cast<T>(0));
    for (size_t t = 0; t <= pos; ++t) {
      for (size_t i = 0; i < kPerHeadDim; ++i) {
        output[{i}] += (attention_scores[t] * V[{i, t}]);
      }
    }
  }

 private:
};

}  // namespace llama2
