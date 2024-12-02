/**
 * @file memory_buffer_cpu.hpp
 * @brief This file defines the memory buffer class for the cpu malloc device.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
#include <memory>
#include <vector>
// Project Headers
#include <device/malloc/memory_buffer_base.hpp>
// Third-party Headers

namespace llama {

class MemoryBufferCPU : public MemoryBuffer {
 public:
  MemoryBufferCPU(const size_t size, const size_t offset = 0)
      : MemoryBuffer(size, offset),
        buffer_(std::shared_ptr<char[]>(new char[size],
                                        std::default_delete<char[]>())) {}
  MemoryBufferCPU(const size_t size, std::shared_ptr<char[]> buffer,
                  const size_t offset = 0)
      : MemoryBuffer(size, offset), buffer_(buffer) {}

  void* GetBuffer() override {
    return static_cast<void*>(buffer_.get() + offset_);
  }

  const void* GetBuffer() const override {
    return static_cast<const void*>(buffer_.get() + offset_);
  }

  std::shared_ptr<MemoryBuffer> Clone(const size_t offset = 0) override {
    return std::make_shared<MemoryBufferCPU>(GetSizeBytes(), buffer_, offset);
  }

  DeviceType GetDeviceType() const override { return DeviceType::CPU; }

 private:
  std::shared_ptr<char[]> buffer_;
};

}  // namespace llama
