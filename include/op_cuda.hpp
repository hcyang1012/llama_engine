/**
 * @file op_cuda.hpp
 * @brief Header for the Various Operations on CUDA.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-02
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
namespace CudaOps {
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
    void LaunchRmsNormKernel(const float* x, const float* weight, size_t size,
                             float* o, cudaStream_t stream = nullptr);

    LaunchRmsNormKernel(x, weight, size, o);
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
    throw std::runtime_error("Not implemented : MatMul");
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
    throw std::runtime_error("Not implemented : RoPE");
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
    throw std::runtime_error("Not implemented : SoftMax");
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

    throw std::runtime_error("Not implemented : Attention");
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
    throw std::runtime_error("Not implemented : ElementwiseAdd");
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
    throw std::runtime_error("Not implemented : SiLU_EWMul");
  }
};

template <typename T>
class ArgMax {
 public:
  static size_t Compute(const Tensor<T>& input) {
    DCHECK_EQ(input.GetShape().GetRank(), 1)
        << "Input tensor should be 1D tensor";
    throw std::runtime_error("Not implemented : ArgMax");
  }

 private:
};

}  // namespace CudaOps

}  // namespace llama
