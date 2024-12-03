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
#include <dtypes.hpp>
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
    Compute(static_cast<const T*>(x.GetData()->GetBuffer()),
            static_cast<const T*>(weight.GetData()->GetBuffer()),
            x.GetShape()[0], static_cast<T*>(out.GetData()->GetBuffer()));
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
    Compute(static_cast<const T*>(weight.GetData()->GetBuffer()),
            static_cast<const T*>(input.GetData()->GetBuffer()),
            weight.GetShape()[0], weight.GetShape()[1],
            static_cast<T*>(out.GetData()->GetBuffer()));
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
  static void Compute(const size_t position, const TransformerConfig& config,
                      Tensor<T>& Q, Tensor<T>& K) {
    DCHECK_EQ(Q.GetShape()[0], config.Dim())
        << "Input tensor should have the same dimension as the config";

    Compute(position, config, static_cast<T*>(Q.GetData()->GetBuffer()),
            static_cast<T*>(K.GetData()->GetBuffer()));
  }

 private:
  static void Compute(const size_t position, const TransformerConfig& config,
                      float* Q, float* K) {
    const size_t kDim = config.Dim();
    const size_t kNumOfHeads = config.NumHeads();
    const size_t kKVDim = config.KVHeadDim();
    const size_t kHeadDim = config.HeadDim();
    const size_t kNumKVHeads = config.NumKVHeads();
    for (int i = 0; i < kNumOfHeads; i++) {
      for (int j = 0; j < kHeadDim; j += 2) {
        float freq = 1.0f / powf(config.Freq(), (float)j / (float)kHeadDim);
        float val = position * freq;
        float fcr = cosf(val);
        float fci = sinf(val);

        const size_t idx = i * kHeadDim + j;
        float q0 = Q[idx];
        float q1 = Q[idx + 1];
        Q[idx] = q0 * fcr - q1 * fci;
        Q[idx + 1] = q0 * fci + q1 * fcr;
        if (i < kNumKVHeads) {
          float k0 = K[idx];
          float k1 = K[idx + 1];
          K[idx] = k0 * fcr - k1 * fci;
          K[idx + 1] = k0 * fci + k1 * fcr;
        }
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
      Compute(static_cast<const T*>(input.GetData()->GetBuffer()),
              static_cast<float*>(output.GetData()->GetBuffer()), kStride);
    } else {
      const size_t kBatchSize = input.GetShape()[0];
      for (size_t batch = 0; batch < kBatchSize; batch++) {
        Compute(static_cast<const T*>(input.GetData()->GetBuffer()) +
                    batch * kStride,
                static_cast<float*>(output.GetData()->GetBuffer()) +
                    batch * kStride,
                kStride);
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
                      const Tensor<T>& V, const TransformerConfig& config,
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

    Tensor<T> attention_scores({pos + 1}, DeviceType::CPU);
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
    std::fill(static_cast<T*>(output.GetData()->GetBuffer()),
              static_cast<T*>(output.GetData()->GetBuffer()) +
                  output.GetShape().GetSize(),
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
    Compute(static_cast<const T*>(left.GetData()->GetBuffer()),
            static_cast<const T*>(right.GetData()->GetBuffer()),
            static_cast<T*>(output.GetData()->GetBuffer()), size);
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
    Compute(static_cast<const T*>(input.GetData()->GetBuffer()),
            static_cast<const T*>(weight.GetData()->GetBuffer()),
            static_cast<T*>(output.GetData()->GetBuffer()), size);
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

template <typename T>
class MultiAttention {
 public:
  static void Compute(const size_t layer, const size_t pos,
                      const TransformerConfig& config, RunState<T>& run_state) {
    const size_t kNumHeads = config.NumHeads();
    const size_t kNumKVHeads = config.NumKVHeads();
    const size_t kKVMul = kNumKVHeads / kNumHeads;
    const size_t kInputEmbedDim = config.Dim();
    const size_t kSeqLen = config.SeqLen();
    const size_t kHiddenDim = config.HiddenDim();

    for (size_t head_idx = 0; head_idx < kNumHeads; ++head_idx) {
      const size_t kKVHeadIdx = head_idx / kKVMul;
      auto Q = run_state.Q(head_idx);
      auto K_layer = run_state.K(layer);
      auto V_layer = run_state.V(layer);

      auto XB = run_state.XB(head_idx);
      ComputeHead(Q, K_layer, V_layer, config, pos, kKVHeadIdx, XB);
    }
  }

 private:
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
  static void ComputeHead(const Tensor<T>& Q, const Tensor<T>& K,
                          const Tensor<T>& V, const TransformerConfig& config,
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

    Tensor<T> attention_scores({pos + 1}, DeviceType::CPU);
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
    std::fill(static_cast<T*>(output.GetData()->GetBuffer()),
              static_cast<T*>(output.GetData()->GetBuffer()) +
                  output.GetShape().GetSize(),
              static_cast<T>(0));
    for (size_t t = 0; t <= pos; ++t) {
      const float a = attention_scores[t];
      for (size_t i = 0; i < kPerHeadDim; ++i) {
        // output[{i}] += (a * V[{i, header_idx, t}]);
        output[{i}] += (a * V.at(i, header_idx, t));
      }
    }
  }
};

}  // namespace CpuOps

}  // namespace llama
