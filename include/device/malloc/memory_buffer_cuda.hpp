/**
 * @file memory_buffer_cuda.hpp
 * @brief This file defines the memory buffer class for the cuda malloc device.
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
#include <dtypes.hpp>
// Third-party Headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
namespace llama {

class MemoryBufferCUDA : public MemoryBuffer {
 public:
  MemoryBufferCUDA(const size_t size, const size_t offset = 0)
      : MemoryBuffer(size, offset),
        kIsOwner(true),
        buffer_(allocate_buffer(size)) {}

  MemoryBufferCUDA(const size_t size, char* buffer, const size_t offset = 0)
      : MemoryBuffer(size, offset), kIsOwner(false), buffer_(buffer) {}

  void* GetBuffer() override { return static_cast<void*>(buffer_ + offset_); }

  const void* GetBuffer() const override {
    return static_cast<const void*>(buffer_ + offset_);
  }

  std::shared_ptr<MemoryBuffer> Clone(const size_t offset = 0) override {
    return std::make_shared<MemoryBufferCUDA>(GetSizeBytes(), buffer_, offset);
  }

  DeviceType GetDeviceType() const override { return DeviceType::CUDA; }

 private:
  char* allocate_buffer(const size_t size) {
    char* buffer;
    cudaMalloc(&buffer, size);
    return buffer;
  }
  const bool kIsOwner = true;
  char* buffer_;
};

}  // namespace llama
