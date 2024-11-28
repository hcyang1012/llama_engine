/**
 * @file allocator_abc.hpp
 * @brief A abstract base class for memory allocation.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-28
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
// Project Headers

// Third-party Headers

namespace llama {
class MemoryAllocator {
 public:
  virtual void* Allocate(const size_t size) = 0;
  virtual void Free(void* ptr) = 0;

  virtual ~MemoryAllocator() = default;

 private:
};
}  // namespace llama
