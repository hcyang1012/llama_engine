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
#include <dtypes.h>

#include <malloc/malloc_abc.hpp>
#include <malloc/malloc_cpu.hpp>
// Third-party Headers

namespace llama {

std::shared_ptr<MemoryAllocator> GetMemoryAllocator(const DeviceType type) {
  switch (type) {
    case DeviceType::CPU: {
      static std::shared_ptr<MemoryAllocatorCPU> allocator =
          std::make_shared<MemoryAllocatorCPU>();
      return allocator;
    }
    default:
      throw std::invalid_argument("Invalid allocation type");
  }
}

}  // namespace llama
