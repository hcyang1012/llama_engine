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

  virtual ~OpSet() = default;

 protected:
  virtual void RmsNormImpl(const void* x, const void* weight, const size_t size,
                           void* out, const std::type_info& type) = 0;

  virtual void MatMulImpl(const void* weight, const void* input, const size_t n,
                          const size_t d, void* out,
                          const std::type_info& type) = 0;

  virtual void RoPEImpl(const size_t position, const Config& config, void* Q,
                        void* K, const std::type_info& type) = 0;
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
