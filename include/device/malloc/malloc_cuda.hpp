/**
 * @file malloc_cuda.hpp
 * @brief A memory allocator for CPU.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-28
 */

#pragma once
#if defined(USE_CUDA)
// C System-Headers

// C++ System-Headers
#include <cstddef>
// Project Headers
#include <device/malloc/malloc_base.hpp>
#include <device/malloc/memory_buffer_cuda.hpp>
// Third-party Headers

namespace llama {
class MemoryAllocatorCUDA : public MemoryAllocator {
 public:
  std::shared_ptr<MemoryBuffer> Allocate(const size_t byte_size) override {
    return std::make_shared<MemoryBufferCUDA>(byte_size);
  }

  std::shared_ptr<MemoryBuffer> Allocate(char* buffer, const size_t byte_size) {
    return std::make_shared<MemoryBufferCUDA>(byte_size, buffer);
  }

  ~MemoryAllocatorCUDA() = default;
};
}  // namespace llama

#endif