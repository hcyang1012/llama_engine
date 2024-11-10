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
#include <memory>
#include <stdexcept>
#include <vector>
// Project Headers

// Third-party Headers

namespace llama2 {

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

 private:
  const std::vector<size_t> dims = {};
  const size_t kRank = 0;
  const size_t kSize = 0;

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
  /// @brief Constructor
  /// @param shape Shape of the tensor
  explicit Tensor(const Shape &shape)
      : data(new T[shape.GetSize()]),
        kDataBytes(sizeof(T) * shape.GetSize()),
        kIsOwner(true),
        shape(shape) {}
  explicit Tensor(const T *data, const Shape &shape)
      : data(const_cast<T *>(data)),
        kDataBytes(sizeof(T) * shape.GetSize()),
        kIsOwner(false),
        shape(shape) {}
  explicit Tensor(const T *data, const size_t data_bytes)
      : data(const_cast<T *>(data)), kDataBytes(data_bytes), kIsOwner(false) {}

  /// @brief Destructor
  ~Tensor() {
    if (kIsOwner) {
      delete[] data;
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
    for (size_t i = 0; i < indices.size(); i++) {
      index = index * shape[i] + indices[i];
    }
    return data[index];
  }

  const Shape &GetShape() const { return shape; }

  void SetShape(const Shape &shape) {
    if (shape.GetSize() != kDataBytes / sizeof(T)) {
      throw std::invalid_argument("Size of the shape does not match the data");
    }
    this->shape = shape;
  }

  const T *GetData() const { return data; }
  T *GetData() { return data; }
  const size_t GetDataBytesSize() const { return kDataBytes; }

  T &operator[](size_t index) { return data[index]; }
  T operator[](size_t index) const { return data[index]; }

 private:
  T *data;
  const size_t kDataBytes;
  const bool kIsOwner;
  Shape shape;
  const std::string kDataType = typeid(T).name();
};

}  // namespace llama2

bool operator==(const llama2::Shape &lhs, const llama2::Shape &rhs) {
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