/**
 * @file device_cpu.hpp
 * @brief An concrete class for CUDA device.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-01
 */

#pragma once

#if defined(USE_CUDA)

// C System-Headers

// C++ System-Headers
#include <memory>
// Project Headers
#include <device/device_base.hpp>
#include <device/malloc/malloc_cuda.hpp>

// Third-party Headers

namespace llama {

class DeviceCUDA : public Device {
 public:
  MemoryAllocator& GetMemoryAllocator() override { return allocator_; }

 private:
  MemoryAllocatorCUDA allocator_;
};
}  // namespace llama

#endif