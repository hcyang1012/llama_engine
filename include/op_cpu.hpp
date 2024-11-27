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
// For numeric_limits
#include <limits>
// Project Headers

#include <config.hpp>
#include <tensor.hpp>

// Third-party Headers

namespace llama {
namespace CpuOps {
template <typename T>
class RmsNorm {
 public:
  static void Compute(const Tensor<T>& x, const Tensor<T>& weight,
                      Tensor<T>& out) {
    CHECK(x.GetShape() == weight.GetShape())
        << "Size of the input tensors should be the same";
    DCHECK_EQ(x.GetShape().GetRank(), 1) << "Input tensor should be 1D tensor";
    DCHECK_EQ(out.GetShape(), x.GetShape())
        << "Output tensor should have the same shape as the input tensor";
    Compute(x.GetData(), weight.GetData(), x.GetShape()[0], out.GetData());
  }

 private:
  static void Compute(const T* x, const T* weight, const size_t size, T* o) {
    DCHECK_GE(size, 0) << "Size should be greater than or equal to 0";
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
  // Weight : (d,n), Input : (n,1) -> Output : (d,1)
  static void Compute(const Tensor<T>& weight, const Tensor<T>& input,
                      Tensor<T>& out) {
    // Shape[0] : number of cols, Shape[1] : number of rows
    DCHECK_EQ(weight.GetShape().GetRank(), 2)
        << "Weight tensor should be 2D tensor";
    DCHECK_LE(input.GetShape().GetRank(), 2)
        << "Input tensor should be 2D tensor or vector";
    if (input.GetShape().GetRank() == 2) {
      DCHECK_EQ(input.GetShape()[1], 1) << "The last dimension should be 1 if "
                                           "the input tensor is 2D tensor";
      DCHECK_EQ(weight.GetShape()[0], input.GetShape()[1])
          << "Inner dimensions should be the same : " << weight.GetShape()[0]
          << " " << input.GetShape()[1];
    } else {
      DCHECK_EQ(weight.GetShape()[0], input.GetShape()[0])
          << "Inner dimensions should be the same : " << weight.GetShape()[0]
          << " " << input.GetShape()[0];
    }
    DCHECK_LE(out.GetShape().GetRank(), 2)
        << "Output tensor should be 2D tensor or vector";

    const auto& kOutShape = out.GetShape();
    if (out.GetShape().GetRank() == 2) {
      DCHECK_EQ(kOutShape[1], weight.GetShape()[1]);
    } else {
      const Shape kExpectedShape({weight.GetShape()[1]});
      DCHECK_EQ(kOutShape, kExpectedShape)
          << "Output tensor should have the shape of " << kExpectedShape;
    }
    Compute(weight.GetData(), input.GetData(), weight.GetShape()[0],
            weight.GetShape()[1], out.GetData());
  }

 private:
  static void Compute(const T* weight, const T* input, const size_t n,
                      const size_t d, T* out) {
    DCHECK_GE(n, 0) << "Size 'n' should be greater than or equal to 0";
    DCHECK_GE(d, 0) << "Size 'd' should be greater than or equal to 0";
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    for (size_t i = 0; i < d; i++) {
      T val = static_cast<T>(0.0f);
      for (size_t j = 0; j < n; j++) {
        val += weight[i * n + j] * input[j];
      }
      out[i] = val;
    }
  }
};

template <typename T>
class RoPE {
 public:
  static void Compute(const size_t position, const Config& config,
                      Tensor<float>& Q, Tensor<float>& K) {
    DCHECK_EQ(Q.GetShape()[0], config.Dim())
        << "Input tensor should have the same dimension as the config";

    Compute(position, config, Q.GetData(), K.GetData());
  }

 private:
  static void Compute(const size_t position, const Config& config, float* Q,
                      float* K) {
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
        float* vec = v == 0 ? Q : K;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }
  }
};

template <typename T>
class SoftMax {
 public:
  static void Compute(const Tensor<T>& input, Tensor<float>& output) {
    DCHECK_EQ(input.GetShape(), output.GetShape())
        << "Input and output tensor should have the same shape";
    DCHECK_LE(input.GetShape().GetRank(), 2)
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

 private:
  static void Compute(const T* input, float* output, const size_t size) {
    DCHECK_GE(size, 0) << "Size should be greater than or equal to 0";
    float max_val = input[0];
    for (size_t i = 1; i < size; i++) {
      max_val = std::max(max_val, static_cast<float>(input[i]));
    }
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) {
      output[i] = expf(static_cast<float>(input[i]) - max_val);
      sum += output[i];
    }
    for (size_t i = 0; i < size; i++) {
      output[i] /= sum;
    }
  }
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
    DCHECK_EQ(Q.GetShape()[0], kPerHeadDim);

    DCHECK_EQ(K.GetShape()[0], kPerHeadDim);
    DCHECK_EQ(K.GetShape()[1], config.NumKVHeads());
    DCHECK_EQ(V.GetShape()[2], config.SeqLen());

    DCHECK_EQ(V.GetShape()[0], kPerHeadDim);
    DCHECK_EQ(V.GetShape()[1], config.NumKVHeads());
    DCHECK_EQ(V.GetShape()[2], config.SeqLen());

    Tensor<T> attention_scores({pos + 1});
    for (size_t t = 0; t <= pos; ++t) {
      float score = 0.0f;
      for (size_t i = 0; i < kPerHeadDim; ++i) {
        score += (Q.at(i) * K.at(i, header_idx, t));
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
      const float a = attention_scores[t];
      for (size_t i = 0; i < kPerHeadDim; ++i) {
        // output[{i}] += (a * V[{i, header_idx, t}]);
        output[{i}] += (a * V.at(i, header_idx, t));
      }
    }
  }

 private:
};

template <typename T>
class ElementwiseAdd {
 public:
  static void Compute(const Tensor<T>& left, const Tensor<T>& right,
                      Tensor<T>& output) {
    DCHECK_EQ(left.GetShape(), right.GetShape())
        << "Input tensors should have the same shape";
    DCHECK_EQ(left.GetShape(), output.GetShape())
        << "Output tensor should have the same shape as the input tensor";

    const size_t size = left.GetShape().GetSize();
    Compute(left.GetData(), right.GetData(), output.GetData(), size);
  }

 private:
  static void Compute(const T* left, const T* right, T* output,
                      const size_t size) {
    DCHECK_GE(size, 0) << "Size should be greater than or equal to 0";
    for (size_t i = 0; i < size; i++) {
      output[i] = left[i] + right[i];
    }
  }
};

template <typename T>
class SiLU_EWMul {
 public:
  static void Compute(const Tensor<T>& input, const Tensor<T>& weight,
                      Tensor<T>& output) {
    DCHECK_EQ(input.GetShape(), weight.GetShape())
        << "Input tensors should have the same shape";
    DCHECK_EQ(input.GetShape(), output.GetShape())
        << "Output tensor should have the same shape as the input tensor";

    const size_t size = input.GetShape().GetSize();
    Compute(input.GetData(), weight.GetData(), output.GetData(), size);
  }

 private:
  static void Compute(const T* input, const T* weight, T* output,
                      const size_t size) {
    DCHECK_GE(size, 0) << "Size should be greater than or equal to 0";
    for (size_t i = 0; i < size; i++) {
#if 0
      // The multiplication order is changed in the SiLU implementation from the
      // code in reference.cpp, so the output is different from the reference
      // code.
      output[i] =
          static_cast<T>(input[i]) * static_cast<T>(weight[i]) *
          static_cast<T>(1.0f / (1.0f + expf(-static_cast<float>(input[i]))));
#else
      float val = static_cast<float>(input[i]);
      val *= (1.0f / (1.0f + expf(-val)));
      val *= static_cast<float>(weight[i]);
      output[i] = static_cast<T>(val);
#endif
    }
  }
};

template <typename T>
class ArgMax {
 public:
  static size_t Compute(const Tensor<T>& input) {
    DCHECK_EQ(input.GetShape().GetRank(), 1)
        << "Input tensor should be 1D tensor";
    const size_t kSize = input.GetShape()[0];
    size_t max_idx = 0;
    T max_val = input[0];
    for (size_t i = 1; i < kSize; i++) {
      if (input[i] > max_val) {
        max_val = input[i];
        max_idx = i;
      }
    }
    return max_idx;
  }

 private:
};

}  // namespace CpuOps

}  // namespace llama
