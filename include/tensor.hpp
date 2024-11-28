/**
 * @file template.hpp
 * @brief Library for tensor operations.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
// Project Headers
#include <malloc/malloc.hpp>

// Third-party Headers
#include <glog/logging.h>

namespace llama {

constexpr int MAX_DIM = 4;  ///< Maximum number of dimensions

/// @brief Shape class for tensor
class Shape {
 public:
  /// @brief Constructor
  /// @param dims Dimensions of the tensor

  Shape(const std::initializer_list<size_t> &dims)
      : dims(dims), kRank(dims.size()), kSize(calc_size()) {
    if (dims.size() > MAX_DIM) {
      throw std::invalid_argument("Number of dimensions(" +
                                  std::to_string(dims.size()) + ") exceeds " +
                                  std::to_string(MAX_DIM));
    }
  }

  /// @brief Get the dimension at the index
  /// @param index Index of the dimension
  /// @return Dimension at the index
  const size_t &operator[](size_t index) const { return dims[index]; }

  /// @brief Get the rank of the tensor
  /// @return Rank of the tensor
  size_t GetRank() const { return kRank; }

  /// @brief Get the size of shape
  /// @return Size of shape
  size_t GetSize() const { return kSize; }

  Shape &operator=(const Shape &other) {
    if (this == &other) {
      return *this;
    }
    dims = other.dims;
    kRank = other.kRank;
    kSize = other.kSize;
    return *this;
  }

 private:
  std::vector<size_t> dims = {};
  size_t kRank = 0;
  size_t kSize = 0;

  size_t calc_size() const {
    size_t size = 1;
    for (auto dim : dims) {
      size *= dim;
    }
    return size;
  }
};

/// @brief Tensor class
/// @tparam T Data type of the tensor
template <typename T>
class Tensor {
 public:
  explicit Tensor(const T *data, const Shape &shape,
                  const DeviceType &device_type)
      : data(const_cast<T *>(data)),
        kDataBytes(sizeof(T) * shape.GetSize()),
        kIsOwner(false),
        shape(shape),
        device_type_(device_type),
        allocator(GetMemoryAllocator(device_type)) {}

  explicit Tensor(const Shape &shape, const DeviceType &type)
      : kDataBytes(sizeof(T) * shape.GetSize()),
        kIsOwner(true),
        shape(shape),
        device_type_(type),
        allocator(GetMemoryAllocator(type)),
        data(reinterpret_cast<T *>(allocator->Allocate(kDataBytes))) {}

  Tensor(const Tensor<T> &other)
      : kDataBytes(other.GetDataBytesSize()),
        kIsOwner(false),
        shape(other.GetShape()),
        device_type_(other.device_type_),
        allocator(GetMemoryAllocator(other.device_type_)),
        data(const_cast<T *>(other.GetData())) {}

  std::string ToString() const {
    std::stringstream ss;
    for (size_t i = 0; i < shape.GetSize(); i++) {
      ss << "[" << i << "]\t" << data[i] << std::endl;
    }
    return ss.str();
  }

  /// @brief Destructor
  ~Tensor() {
    if (kIsOwner) {
      allocator->Free(data);
    }
  }

  /// @brief Get the data at the index
  /// @param index Index of the data
  /// @return Data at the index
  T &operator[](const std::vector<size_t> &indices) {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
      index = index * shape[i] + indices[i];
    }
    return data[index];
  }
  const T &operator[](const std::vector<size_t> &indices) const {
    size_t index = 0;
    // The first element of the shape is the least significant dimension
    // For example, for a 2D tensor, the shape is {3, 4}
    // [2,3] = 3 * 3 + 2 = 11
    for (int i = indices.size() - 1; i >= 0; i--) {
      index = index * shape[i] + indices[i];
    }
    return data[index];
  }

  const T &at(const size_t i0) const {
    DCHECK_EQ(shape.GetRank(), 1) << "Tensor should be 1D tensor";
    DCHECK_LT(i0, shape[0]) << "Index[0] out of range" << i0 << " " << shape[0];
    return data[i0];
  }

  const T &at(const size_t i0, const size_t i1) const {
    DCHECK_EQ(shape.GetRank(), 2) << "Tensor should be 2D tensor";
    DCHECK_LT(i0, shape[0]) << "Index[0] out of range" << i0 << " " << shape[0];
    DCHECK_LT(i1, shape[1]) << "Index[1] out of range" << i1 << " " << shape[1];
    return data[i1 * shape[0] + i0];
  }

  const T &at(const size_t i0, const size_t i1, const size_t i2) const {
    DCHECK_EQ(shape.GetRank(), 3) << "Tensor should be 3D tensor";
    DCHECK_LT(i0, shape[0]) << "Index[0] out of range" << i0 << " " << shape[0];
    DCHECK_LT(i1, shape[1]) << "Index[1] out of range" << i1 << " " << shape[1];
    DCHECK_LT(i2, shape[2]) << "Index[2] out of range" << i2 << " " << shape[2];
    return data[i2 * (shape[0] * shape[1]) + i1 * shape[0] + i0];
  }

  const Shape &GetShape() const { return shape; }

  void SetShape(const Shape &shape) {
    if (shape.GetSize() != kDataBytes / sizeof(T)) {
      throw std::invalid_argument("Size of the shape does not match the data");
    }
    this->shape = shape;
  }

  Tensor<T> ReShape(const Shape &shape) const {
    if (shape.GetSize() != kDataBytes / sizeof(T)) {
      throw std::invalid_argument("Size of the shape does not match the data");
    }
    return Tensor<T>(data, shape, device_type_);
  }

  const T *GetData() const { return data; }
  T *GetData() { return data; }
  const size_t GetDataBytesSize() const { return kDataBytes; }

  T &operator[](size_t index) { return data[index]; }
  T operator[](size_t index) const { return data[index]; }

  Tensor<T> &operator=(const Tensor<T> &other) {
    if (this == &other) {
      return *this;
    }

    if (kIsOwner) {
      allocator->Free(data);
    }

    kIsOwner = false;
    allocator = other.allocator;
    kDataBytes = other.kDataBytes;
    shape = other.shape;
    device_type_ = other.device_type_;
    data = const_cast<T *>(other.data);

    return *this;
  }

 private:
  size_t kDataBytes;
  bool kIsOwner;
  Shape shape;
  const DeviceType device_type_;
  std::shared_ptr<MemoryAllocator> allocator;

  T *data;
};

std::ostream &operator<<(std::ostream &os, const llama::Shape &shape) {
  os << "(";
  for (size_t i = 0; i < shape.GetRank(); i++) {
    os << shape[i];
    if (i != shape.GetRank() - 1) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

}  // namespace llama

bool operator==(const llama::Shape &lhs, const llama::Shape &rhs) {
  if (lhs.GetRank() != rhs.GetRank()) {
    return false;
  }
  for (size_t i = 0; i < lhs.GetRank(); i++) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}
