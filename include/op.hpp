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

#include <op_cpu.hpp>
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
    RmsNormImpl(x.GetData(), weight.GetData(), x.GetShape()[0], out.GetData(),
                typeid(T));
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

    MatMulImpl(weight.GetData(), input.GetData(), weight.GetShape()[0],
               weight.GetShape()[1], out.GetData(), typeid(T));
  }

  template <typename T>
  void RoPE(const size_t position, const Config& config, Tensor<float>& Q,
            Tensor<float>& K) {
    CHECK_EQ(Q.GetShape()[0], config.Dim())
        << "Input tensor should have the same dimension as the config";

    RoPEImpl(position, config, Q.GetData(), K.GetData(), typeid(T));
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

    if (input.GetShape().GetRank() == 1) {
      SoftMaxImpl(input.GetData(), output.GetData(), kStride, typeid(T));
    } else {
      const size_t kBatchSize = input.GetShape()[0];
      for (size_t batch = 0; batch < kBatchSize; batch++) {
        SoftMaxImpl(input.GetData() + batch * kStride,
                    output.GetData() + batch * kStride, kStride, typeid(T));
      }
    }
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
    ElementwiseAddImpl(&left, &right, &output, size);
  }

  template <typename T>
  void SiLU_EWMul(const Tensor<T>& input, const Tensor<T>& weight,
                  Tensor<T>& output) {
    DCHECK_EQ(input.GetShape(), weight.GetShape())
        << "Input tensors should have the same shape";
    DCHECK_EQ(input.GetShape(), output.GetShape())
        << "Output tensor should have the same shape as the input tensor";

    const size_t size = input.GetShape().GetSize();
    SiLU_EWMulImpl(&input, &weight, &output, size);
  }

  template <typename T>
  size_t ArgMax(const Tensor<T>& input) {
    DCHECK_EQ(input.GetShape().GetRank(), 1)
        << "Input tensor should be 1D tensor";
    return ArgMaxImpl(&input, typeid(T));
  }

  virtual ~OpSet() = default;

 protected:
  virtual void RmsNormImpl(const void* x, const void* weight, const size_t size,
                           void* out, const std::type_info& type) = 0;

  virtual void MatMulImpl(const void* weight, const void* input, const size_t n,
                          const size_t d, void* out,
                          const std::type_info& type) = 0;

  virtual void RoPEImpl(const size_t position, const Config& config, void* Q,
                        void* K, const std::type_info& type) = 0;

  virtual void SoftMaxImpl(const void* input, void* output, const size_t size,
                           const std::type_info& type) = 0;

  virtual void AttentionImpl(const void* Q, const void* K, const void* V,
                             const Config& config, const size_t pos,
                             const size_t header_idx, void* output,
                             const std::type_info& type) = 0;

  virtual void ElementwiseAddImpl(const void* left, const void* right,
                                  void* output, const size_t size) = 0;

  virtual void SiLU_EWMulImpl(const void* input, const void* weight,
                              void* output, const size_t size) = 0;

  virtual size_t ArgMaxImpl(const void* input, const std::type_info& type) = 0;
};

class OpSetCpu : public OpSet {
 public:
  OpSetCpu() {}
  void RmsNormImpl(const void* x, const void* weight, const size_t size,
                   void* out, const std::type_info& type) override {
    if (type == typeid(float)) {
      OPSetCpu::RmsNorm<float>::Compute(static_cast<const float*>(x),
                                        static_cast<const float*>(weight), size,
                                        static_cast<float*>(out));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void MatMulImpl(const void* weight, const void* input, const size_t n,
                  const size_t d, void* out,
                  const std::type_info& type) override {
    if (type == typeid(float)) {
      OPSetCpu::MatMul<float>::Compute(static_cast<const float*>(weight),
                                       static_cast<const float*>(input), n, d,
                                       static_cast<float*>(out));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void RoPEImpl(const size_t position, const Config& config, void* Q, void* K,
                const std::type_info& type) override {
    if (type == typeid(float)) {
      OPSetCpu::RoPE<float>::Compute(position, config, static_cast<float*>(Q),
                                     static_cast<float*>(K));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void SoftMaxImpl(const void* input, void* output, const size_t size,
                   const std::type_info& type) override {
    if (type == typeid(float)) {
      OPSetCpu::SoftMax<float>::Compute(static_cast<const float*>(input),
                                        static_cast<float*>(output), size);
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void AttentionImpl(const void* Q, const void* K, const void* V,
                     const Config& config, const size_t pos,
                     const size_t header_idx, void* output,
                     const std::type_info& type) override {
    if (type == typeid(float)) {
      OPSetCpu::Attention<float>::Compute(*static_cast<const Tensor<float>*>(Q),
                                          *static_cast<const Tensor<float>*>(K),
                                          *static_cast<const Tensor<float>*>(V),
                                          config, pos, header_idx,
                                          *static_cast<Tensor<float>*>(output));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void ElementwiseAddImpl(const void* left, const void* right, void* output,
                          const size_t size) override {
    OPSetCpu::ElementwiseAdd<float>::Compute(
        *static_cast<const Tensor<float>*>(left),
        *static_cast<const Tensor<float>*>(right),
        *static_cast<Tensor<float>*>(output));
  }

  void SiLU_EWMulImpl(const void* input, const void* weight, void* output,
                      const size_t size) override {
    OPSetCpu::SiLU_EWMul<float>::Compute(
        *static_cast<const Tensor<float>*>(input),
        *static_cast<const Tensor<float>*>(weight),
        *static_cast<Tensor<float>*>(output));
  }

  size_t ArgMaxImpl(const void* input, const std::type_info& type) override {
    if (type == typeid(float)) {
      return OPSetCpu::ArgMax<float>::Compute(
          *static_cast<const Tensor<float>*>(input));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

 private:
};

std::unique_ptr<OpSet> CreateOpSet(OpSet::OpType type) {
  switch (type) {
    case OpSet::OpType::CPU:
      return std::make_unique<OpSetCpu>();
    default:
      LOG(FATAL) << "Unsupported OpType";
  }
}

}  // namespace llama
