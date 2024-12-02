/**
 * @file device_cpu.hpp
 * @brief An concrete class for CPU device.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-01
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <memory>
// Project Headers
#include <device/device_base.hpp>
#include <device/malloc/malloc_cpu.hpp>

// Third-party Headers

namespace llama {

class DeviceCPU : public Device {
 public:
  MemoryAllocator& GetMemoryAllocator() override { return allocator_; }
  MemcpyBase& GetMemcpy() override { return memcpy_; }

 private:
  MemoryAllocatorCPU allocator_;
  MemcpyCPU memcpy_;
};
}  // namespace llama
