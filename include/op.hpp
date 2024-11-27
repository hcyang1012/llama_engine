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

 protected:
  virtual void RmsNormImpl(const void* x, const void* weight, const size_t size,
                           void* out, const std::type_info& type) = 0;
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
