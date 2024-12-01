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
#include <device/malloc/malloc_abc.hpp>
// Third-party Headers

namespace llama {
class MemoryAllocatorCPU : public MemoryAllocator {
 public:
  bool Allocate(void** dst, const size_t size) override {
    *dst = new char[size];
    return true;
  }
  bool Free(void* ptr) override {
    delete[] static_cast<char*>(ptr);
    return true;
  }

  ~MemoryAllocatorCPU() = default;
};
}  // namespace llama
