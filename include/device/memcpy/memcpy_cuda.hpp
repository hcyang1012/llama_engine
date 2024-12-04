/**
 * @file memcpy_cpu.hpp
 * @brief This file is a memory copy operation for CPU.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-02
 */

#pragma once
#if defined(USE_CUDA)
// C System-Headers

// C++ System-Headers
#include <algorithm>
// Project Headers
#include <device/memcpy/memcpy_base.hpp>
// Third-party Headers
#include <glog/logging.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
namespace llama {
class MemcpyCUDA : public MemcpyBase {
 public:
  void Copy(MemoryBuffer &dst, const MemoryBuffer &src,
            const size_t bytes_size) override {
    const auto kSrcDeviceType = src.GetDeviceType();
    const auto kDstDeviceType = dst.GetDeviceType();
    if (kDstDeviceType == DeviceType::CPU &&
        kSrcDeviceType == DeviceType::CUDA) {
      cudaMemcpy(dst.GetBuffer(), src.GetBuffer(), bytes_size,
                 cudaMemcpyDeviceToHost);
    } else if (kDstDeviceType == DeviceType::CUDA &&
               kSrcDeviceType == DeviceType::CPU) {
      cudaMemcpy(dst.GetBuffer(), src.GetBuffer(), bytes_size,
                 cudaMemcpyHostToDevice);
    } else if (kDstDeviceType == DeviceType::CUDA &&
               kSrcDeviceType == DeviceType::CUDA) {
      LOG(FATAL) << "Unsupported device type from CUDA to CUDA";
    } else {
      LOG(FATAL) << "Unsupported device type from CPU to CPU";
    }
  }
  void Copy(MemoryBuffer &dst, const void *src, const size_t size) override {
    DCHECK_EQ(dst.GetSizeBytes(), size)
        << "Size of the destination and source buffer does not match";
    DCHECK_EQ(dst.GetDeviceType(), DeviceType::CUDA)
        << "Destination buffer should be CUDA device";

    cudaMemcpy(dst.GetBuffer(), src, size, cudaMemcpyHostToDevice);
  }
};
}  // namespace llama

#endif