/**
 * @file op.hpp
 * @brief Header for the Various Operations
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-11
 */

#pragma once

// C System-Headers

// C++ System-Headers

#include <memory>

// Project Headers

#include <dtypes.hpp>
#include <opset.hpp>
#include <opset_cpu.hpp>
#include <tensor.hpp>
// Third-party Headers
#include <glog/logging.h>
namespace llama {

std::unique_ptr<OpSet> CreateOpSet(const DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return std::make_unique<OpSetCpu>();
    default:
      LOG(FATAL) << "Unsupported OpType";
  }
}

}  // namespace llama
