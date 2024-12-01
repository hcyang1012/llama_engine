/**
 * @file device.hpp
 * @brief A class for device factory.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-01
 */

#pragma once

// C System-Headers

// C++ System-Headers

// Project Headers
#include <device/device_cpu.hpp>
#include <device/device_cuda.hpp>
#include <dtypes.hpp>
// Third-party Headers

namespace llama {

class DeviceFactory {
 public:
  static Device& GetDevice(const DeviceType type) {
    switch (type) {
      case DeviceType::CPU:
        static DeviceCPU cpu_device;
        return cpu_device;
#if defined(USE_CUDA)
      case DeviceType::CUDA:
        static DeviceCUDA cuda_device;
        return cuda_device;
#endif
      default:
        throw std::invalid_argument("Invalid device type");
    }
  }
};
}  // namespace llama
