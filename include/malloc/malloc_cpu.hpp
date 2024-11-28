/**
 * @file malloc_cpu.hpp
 * @brief A memory allocator for CPU.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-28
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
// Project Headers
#include <malloc/malloc_abc.hpp>
// Third-party Headers

namespace llama {
class MemoryAllocatorCPU : public MemoryAllocator {
 public:
  void* Allocate(const size_t size) override { return new char[size]; }
  void Free(void* ptr) override { delete[] static_cast<char*>(ptr); }

  ~MemoryAllocatorCPU() = default;
};
}  // namespace llama
