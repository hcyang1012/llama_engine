/**
 * @file opset_cpu.hpp
 * @brief Header for the Various Operations on CPU.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-11
 */

#pragma once

// C System-Headers

// C++ System-Headers

#include <memory>

// Project Headers

#include <op_cpu.hpp>
#include <opset.hpp>
#include <tensor.hpp>
// Third-party Headers
#include <glog/logging.h>
namespace llama {

class OpSetCpu : public OpSet {
 public:
  OpSetCpu() {}

  DeviceType GetDeviceType() const override { return DeviceType::CPU; }

  void RmsNormImpl(const void* x, const void* weight, void* out,
                   const std::type_info& type) override {
    if (type == typeid(float)) {
      CpuOps::RmsNorm<float>::Compute(
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
      CpuOps::MatMul<float>::Compute(*static_cast<const Tensor<float>*>(weight),
                                     *static_cast<const Tensor<float>*>(input),
                                     *static_cast<Tensor<float>*>(out));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void RoPEImpl(const size_t position, const Config& config, void* Q, void* K,
                const std::type_info& type) override {
    if (type == typeid(float)) {
      CpuOps::RoPE<float>::Compute(position, config,
                                   *static_cast<Tensor<float>*>(Q),
                                   *static_cast<Tensor<float>*>(K));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }
  void SoftMaxImpl(const void* input, void* output,
                   const std::type_info& type) override {
    if (type == typeid(float)) {
      CpuOps::SoftMax<float>::Compute(*static_cast<const Tensor<float>*>(input),
                                      *static_cast<Tensor<float>*>(output));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  void AttentionImpl(const void* Q, const void* K, const void* V,
                     const Config& config, const size_t pos,
                     const size_t header_idx, void* output,
                     const std::type_info& type) override {
    if (type == typeid(float)) {
      CpuOps::Attention<float>::Compute(*static_cast<const Tensor<float>*>(Q),
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
      CpuOps::ElementwiseAdd<float>::Compute(
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
      CpuOps::SiLU_EWMul<float>::Compute(
          *static_cast<const Tensor<float>*>(input),
          *static_cast<const Tensor<float>*>(weight),
          *static_cast<Tensor<float>*>(output));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

  size_t ArgMaxImpl(const void* input, const std::type_info& type) override {
    if (type == typeid(float)) {
      return CpuOps::ArgMax<float>::Compute(
          *static_cast<const Tensor<float>*>(input));
    } else {
      LOG(FATAL) << "Unsupported data type";
    }
  }

 private:
};

}  // namespace llama