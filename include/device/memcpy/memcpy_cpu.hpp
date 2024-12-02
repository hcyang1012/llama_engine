/**
 * @file memcpy_cpu.hpp
 * @brief This file is a memory copy operation for CPU.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-02
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <algorithm>
// Project Headers
#include <device/memcpy/memcpy_base.hpp>
// Third-party Headers
#include <glog/logging.h>
namespace llama {
class MemcpyCPU : public MemcpyBase {
 public:
  void Copy(MemoryBuffer &dst, const MemoryBuffer &src) {
    DCHECK_EQ(dst.GetSizeBytes(), src.GetSizeBytes())
        << "Size of the destination and source buffer does not match";

    DCHECK_EQ(dst.GetDeviceType(), DeviceType::CPU)
        << "Destination buffer is not on CPU";
    DCHECK_EQ(src.GetDeviceType(), DeviceType::CPU)
        << "Source buffer is not on CPU";

    std::copy(static_cast<const char *>(src.GetBuffer()),
              static_cast<const char *>(src.GetBuffer()) + src.GetSizeBytes(),
              static_cast<char *>(dst.GetBuffer()));
  }

  void Copy(MemoryBuffer &dst, const void *src, const size_t size) {
    DCHECK_EQ(dst.GetDeviceType(), DeviceType::CPU)
        << "Destination buffer is not on CPU";

    std::copy(static_cast<const char *>(src),
              static_cast<const char *>(src) + size,
              static_cast<char *>(dst.GetBuffer()));
  }
};
}  // namespace llama
