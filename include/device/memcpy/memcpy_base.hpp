/**
 * @file memcpy_base.hpp
 * @brief This file is a base class for memory copy operations.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-02
 */

#pragma once

// C System-Headers

// C++ System-Headers

// Project Headers
#include <device/malloc/memory_buffer_base.hpp>
// Third-party Headers
#include <glog/logging.h>

namespace llama {
class MemcpyBase {
 public:
  virtual void Copy(MemoryBuffer& dst, const MemoryBuffer& src,
                    const size_t bytes_size) = 0;
  virtual void Copy(MemoryBuffer& dst, const void* src, const size_t size) = 0;

  virtual void Copy(MemoryBuffer& dst, const MemoryBuffer& src) {
    DCHECK_EQ(dst.GetSizeBytes(), src.GetSizeBytes())
        << "Size of the destination and source buffer does not match";
    Copy(dst, src, src.GetSizeBytes());
  }
};
}  // namespace llama
