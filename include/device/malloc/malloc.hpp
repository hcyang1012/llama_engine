/**
 * @file malloc.hpp
 * @brief
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-28
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
#include <memory>
#include <stdexcept>
// Project Headers
#include <device/malloc/malloc_base.hpp>
#include <device/malloc/malloc_cpu.hpp>
#include <device/malloc/malloc_cuda.hpp>
#include <dtypes.hpp>
// Third-party Headers

namespace llama {

MemoryAllocator& GetMemoryAllocator(const DeviceType type) {
  switch (type) {
    case DeviceType::CPU: {
      static MemoryAllocatorCPU allocator;
      return allocator;
    }
#if defined(__CUDACC__)
    case DeviceType::CUDA: {
      static MemoryAllocatorCUDA allocator;
      return allocator;
    }
#endif
    default:
      throw std::invalid_argument("Invalid allocation type");
  }
}

}  // namespace llama
