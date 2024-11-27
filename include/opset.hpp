/**
 * @file op.hpp
 * @brief Header for the Various Operations.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-11
 */

#pragma once

// C System-Headers

// C++ System-Headers

#include <memory>

// Project Headers

#include <config.hpp>
#include <tensor.hpp>
// Third-party Headers
#include <glog/logging.h>
namespace llama {

// Virtual class for RMSNorm
class OpSet {
 public:
  enum class OpType { CPU };

  template <typename T>
  void RmsNorm(const Tensor<T>& x, const Tensor<T>& weight, Tensor<T>& out) {
    CHECK(x.GetShape() == weight.GetShape())
        << "Size of the input tensors should be the same";
    DCHECK_EQ(x.GetShape().GetRank(), 1) << "Input tensor should be 1D tensor";
    DCHECK_EQ(out.GetShape(), x.GetShape())
        << "Output tensor should have the same shape as the input tensor";
    RmsNormImpl(&x, &weight, &out, typeid(T));
  }

  template <typename T>
  void MatMul(const Tensor<T>& weight, const Tensor<T>& input, Tensor<T>& out) {
    CHECK(input.GetShape().GetRank() <= 2)
        << "Input tensor should be 2D tensor or vector";
    CHECK(weight.GetShape().GetRank() == 2)
        << "Weight tensor should be 2D tensor";
    CHECK(out.GetShape().GetRank() <= 2)
        << "Output tensor should be 2D tensor or vector";

    const auto& kOutShape = out.GetShape();
    if (out.GetShape().GetRank() == 2) {
      CHECK_EQ(kOutShape[1], weight.GetShape()[1]);
    } else {
      const Shape kExpectedShape({weight.GetShape()[1]});
      CHECK_EQ(kOutShape, kExpectedShape)
          << "Output tensor should have the shape of " << kExpectedShape;
    }

    MatMulImpl(&weight, &input, &out, typeid(T));
  }

  template <typename T>
  void RoPE(const size_t position, const Config& config, Tensor<float>& Q,
            Tensor<float>& K) {
    CHECK_EQ(Q.GetShape()[0], config.Dim())
        << "Input tensor should have the same dimension as the config";

    RoPEImpl(position, config, &Q, &K, typeid(T));
  }

  template <typename T>
  void SoftMax(const Tensor<T>& input, Tensor<T>& output) {
    CHECK(input.GetShape() == output.GetShape())
        << "Input and output tensor should have the same shape";
    CHECK_LE(input.GetShape().GetRank(), 2)
        << "Input tensor should be 1D or 2D tensor";

    const size_t kStride = input.GetShape().GetRank() == 1
                               ? input.GetShape()[0]
                               : input.GetShape()[1];
    SoftMaxImpl(&input, &output, typeid(T));
  }

  template <typename T>
  void Attention(const Tensor<T>& Q, const Tensor<T>& K, const Tensor<T>& V,
                 const Config& config, const size_t pos,
                 const size_t header_idx, Tensor<T>& output) {
    const size_t kPerHeadDim = config.Dim() / config.NumHeads();
    const size_t kKVHeadDim =
        (config.Dim() * config.NumKVHeads()) / config.NumHeads();
    CHECK_EQ(Q.GetShape()[0], kPerHeadDim);

    CHECK_EQ(K.GetShape()[0], kPerHeadDim);
    CHECK_EQ(K.GetShape()[1], config.NumKVHeads());
    CHECK_EQ(V.GetShape()[2], config.SeqLen());

    CHECK_EQ(V.GetShape()[0], kPerHeadDim);
    CHECK_EQ(V.GetShape()[1], config.NumKVHeads());
    CHECK_EQ(V.GetShape()[2], config.SeqLen());

    AttentionImpl(&Q, &K, &V, config, pos, header_idx, &output, typeid(T));
  }

  template <typename T>
  void ElementwiseAdd(const Tensor<T>& left, const Tensor<T>& right,
                      Tensor<T>& output) {
    CHECK(left.GetShape() == right.GetShape())
        << "Input tensors should have the same shape";
    CHECK(left.GetShape() == output.GetShape())
        << "Output tensor should have the same shape as the input tensor";

    const size_t size = left.GetShape().GetSize();
    ElementwiseAddImpl(&left, &right, &output, size, typeid(T));
  }

  template <typename T>
  void SiLU_EWMul(const Tensor<T>& input, const Tensor<T>& weight,
                  Tensor<T>& output) {
    DCHECK_EQ(input.GetShape(), weight.GetShape())
        << "Input tensors should have the same shape";
    DCHECK_EQ(input.GetShape(), output.GetShape())
        << "Output tensor should have the same shape as the input tensor";

    const size_t size = input.GetShape().GetSize();
    SiLU_EWMulImpl(&input, &weight, &output, size, typeid(T));
  }

  template <typename T>
  size_t ArgMax(const Tensor<T>& input) {
    DCHECK_EQ(input.GetShape().GetRank(), 1)
        << "Input tensor should be 1D tensor";
    return ArgMaxImpl(&input, typeid(T));
  }

  virtual ~OpSet() = default;

 protected:
  virtual void RmsNormImpl(const void* x, const void* weight, void* out,
                           const std::type_info& type) = 0;

  virtual void MatMulImpl(const void* weight, const void* input, void* out,
                          const std::type_info& type) = 0;

  virtual void RoPEImpl(const size_t position, const Config& config, void* Q,
                        void* K, const std::type_info& type) = 0;

  virtual void SoftMaxImpl(const void* input, void* output,
                           const std::type_info& type) = 0;

  virtual void AttentionImpl(const void* Q, const void* K, const void* V,
                             const Config& config, const size_t pos,
                             const size_t header_idx, void* output,
                             const std::type_info& type) = 0;

  virtual void ElementwiseAddImpl(const void* left, const void* right,
                                  void* output, const size_t size,
                                  const std::type_info& type) = 0;

  virtual void SiLU_EWMulImpl(const void* input, const void* weight,
                              void* output, const size_t size,
                              const std::type_info& type) = 0;

  virtual size_t ArgMaxImpl(const void* input, const std::type_info& type) = 0;
};

}  // namespace llama
