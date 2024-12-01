/**
 * @file device_base.hpp
 * @brief An abstract base class for device.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-01
 */

#pragma once

// C System-Headers

// C++ System-Headers

// Project Headers
#include <device/malloc/malloc.hpp>
// Third-party Headers

namespace llama {

class Device {
 public:
  virtual MemoryAllocator& GetMemoryAllocator() = 0;
};
}  // namespace llama
