/**
 * @file opset_cuda.hpp
 * @brief Header for the Various Operations on CUDA.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-11
 */

#pragma once

#if defined(USE_CUDA)

// C System-Headers

// C++ System-Headers

#include <memory>

// Project Headers

#include <op_cuda.hpp>
#include <opset.hpp>
#include <tensor.hpp>
// Third-party Headers
#include <glog/logging.h>
namespace llama {

class OpSetCuda : public OpSet {
 public:
  OpSetCuda() {}

  DeviceType GetDeviceType() const override { return DeviceType::CUDA; }

  void RmsNormImpl(const void* x, const void* weight, void* out,
                   const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::RmsNorm<float>::Compute(
          *static_cast<const Tensor<float>*>(x),
          *static_cast<const Tensor<float>*>(weight),
          *static_cast<Tensor<float>*>(out));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void MatMulImpl(const void* weight, const void* input, void* out,
                  const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::MatMul<float>::Compute(
          *static_cast<const Tensor<float>*>(weight),
          *static_cast<const Tensor<float>*>(input),
          *static_cast<Tensor<float>*>(out));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void RoPEImpl(const size_t position, const TransformerConfig& config, void* Q,
                void* K, const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::RoPE<float>::Compute(position, config,
                                    *static_cast<Tensor<float>*>(Q),
                                    *static_cast<Tensor<float>*>(K));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }
  void SoftMaxImpl(const void* input, void* output,
                   const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::SoftMax<float>::Compute(
          *static_cast<const Tensor<float>*>(input),
          *static_cast<Tensor<float>*>(output));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void AttentionImpl(const void* Q, const void* K, const void* V,
                     const TransformerConfig& config, const size_t pos,
                     const size_t header_idx, void* output,
                     const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::Attention<float>::Compute(*static_cast<const Tensor<float>*>(Q),
                                         *static_cast<const Tensor<float>*>(K),
                                         *static_cast<const Tensor<float>*>(V),
                                         config, pos, header_idx,
                                         *static_cast<Tensor<float>*>(output));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void ElementwiseAddImpl(const void* left, const void* right, void* output,
                          const size_t size,
                          const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::ElementwiseAdd<float>::Compute(
          *static_cast<const Tensor<float>*>(left),
          *static_cast<const Tensor<float>*>(right),
          *static_cast<Tensor<float>*>(output));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void SiLU_EWMulImpl(const void* input, const void* weight, void* output,
                      const size_t size, const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::SiLU_EWMul<float>::Compute(
          *static_cast<const Tensor<float>*>(input),
          *static_cast<const Tensor<float>*>(weight),
          *static_cast<Tensor<float>*>(output));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  size_t ArgMaxImpl(const void* input, const std::type_info& type) override {
    if (type == typeid(float)) {
      return CudaOps::ArgMax<float>::Compute(
          *static_cast<const Tensor<float>*>(input));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void MultiAttentionImpl(const size_t layer, const size_t pos,
                          const TransformerConfig& config, void* run_state,
                          const std::type_info& type) override {
    if (type == typeid(float)) {
      CudaOps::MultiAttention<float>::Compute(
          layer, pos, config, *static_cast<RunState<float>*>(run_state));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

 private:
};

}  // namespace llama

#endif  // #if defined(USE_CUDA)