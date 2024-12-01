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
// Third-party Headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace llama {
class MemoryAllocatorCUDA : public MemoryAllocator {
 public:
  bool Allocate(void** dst, const size_t size) override {
    return cudaMalloc(dst, size) == cudaSuccess;
  }
  bool Free(void* ptr) override { return cudaFree(ptr) == cudaSuccess; }

  ~MemoryAllocatorCUDA() = default;
};
}  // namespace llama

#endif