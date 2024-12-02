/**
 * @file memory_buffer.hpp
 * @brief This file defines the memory buffer class for the malloc device.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
#include <memory>
// Project Headers
#include <dtypes.hpp>
// Third-party Headers

namespace llama {

class MemoryBuffer {
 public:
  MemoryBuffer() = delete;
  virtual ~MemoryBuffer() = default;
  MemoryBuffer(const size_t size, const size_t offset = 0)
      : byte_size_(size), offset_(offset) {}

  MemoryBuffer(const MemoryBuffer&) = delete;
  MemoryBuffer& operator=(const MemoryBuffer&) = delete;

  const size_t GetSizeBytes() const { return byte_size_; }
  virtual void* GetBuffer() = 0;
  virtual const void* GetBuffer() const = 0;

  virtual std::shared_ptr<MemoryBuffer> Clone(const size_t offset = 0) = 0;

  virtual DeviceType GetDeviceType() const = 0;

 protected:
  const size_t offset_;

 private:
  const size_t byte_size_;
};

}  // namespace llama
