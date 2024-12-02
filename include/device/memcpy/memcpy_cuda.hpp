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
  void Copy(MemoryBuffer &dst, const MemoryBuffer &src) {
    DCHECK_EQ(dst.GetSizeBytes(), src.GetSizeBytes())
        << "Size of the destination and source buffer does not match";

    if (dst.GetDeviceType() == DeviceType::CPU &&
        src.GetDeviceType() == DeviceType::CUDA) {
      cudaMemcpy(dst.GetBuffer(), src.GetBuffer(), src.GetSizeBytes(),
                 cudaMemcpyDeviceToHost);
    } else if (dst.GetDeviceType() == DeviceType::CUDA &&
               src.GetDeviceType() == DeviceType::CPU) {
      cudaMemcpy(dst.GetBuffer(), src.GetBuffer(), src.GetSizeBytes(),
                 cudaMemcpyHostToDevice);
    } else {
      LOG(FATAL) << "Unsupported device type from CPU to CPU";
    }
  }

  void Copy(MemoryBuffer &dst, const void *src, const size_t size) {
    DCHECK_EQ(dst.GetSizeBytes(), size)
        << "Size of the destination and source buffer does not match";
    DCHECK_EQ(dst.GetDeviceType(), DeviceType::CUDA)
        << "Destination buffer should be CUDA device";

    cudaMemcpy(dst.GetBuffer(), src, size, cudaMemcpyHostToDevice);
  }
};
}  // namespace llama

#endif