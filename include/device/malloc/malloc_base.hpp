/**
 * @file allocator_base.hpp
 * @brief An abstract base class for memory allocation.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-28
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
// Project Headers
#include <device/malloc/memory_buffer_base.hpp>
// Third-party Headers

namespace llama {
class MemoryAllocator {
 public:
  virtual std::shared_ptr<MemoryBuffer> Allocate(const size_t size) = 0;

  virtual ~MemoryAllocator() = default;

 private:
};
}  // namespace llama
