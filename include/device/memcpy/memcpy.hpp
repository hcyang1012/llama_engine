/**
 * @file memcpy.hpp
 * @brief
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-28
 */

#pragma once

// C System-Headers

// C++ System-Headers

// Project Headers
#include <device/memcpy/memcpy_base.hpp>
#include <device/memcpy/memcpy_cpu.hpp>
#include <device/memcpy/memcpy_cuda.hpp>
#include <dtypes.hpp>
// Third-party Headers

namespace llama {

MemcpyBase& GetMemcpy(const DeviceType type) {
  switch (type) {
    case DeviceType::CPU: {
      static MemcpyCPU memcpy_cpu;
      return memcpy_cpu;
    }
#if defined(USE_CUDA)
    case DeviceType::CUDA: {
      static MemcpyCUDA memcpy_cuda;
      return memcpy_cuda;
    }
#endif
    default:
      throw std::invalid_argument("Invalid memcpy type");
  }
}
}  // namespace llama
