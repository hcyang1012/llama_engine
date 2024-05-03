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

constexpr int MAX_DIM = 4; ///< Maximum number of dimensions

/// @brief Shape class for tensor
class Shape {
public:
  /// @brief Constructor
  /// @param dims Dimensions of the tensor. The last dimension is the innermost
  /// @throw std::invalid_argument if the number of dimensions exceeds MAX_DIM
  Shape(const std::vector<size_t> &dims)
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
  const std::vector<size_t> dims;
  const size_t kRank;
  const size_t kSize;

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
template <typename T> class Tensor {
public:
  /// @brief Constructor
  /// @param shape Shape of the tensor
  Tensor(const Shape &shape) : shape(shape), data(shape.GetSize()) {}

  /// @brief Get the data at the index
  /// @param index Index of the data
  /// @return Data at the index
  T &operator[](const std::vector<size_t> &indices) {}

  /// @brief Get the data at the index
  /// @param indices Indices of the data. The last index is the innermost
  /// @return Data at the index
  const T &operator[](const std::vector<size_t> &indices) const {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); i++) {
      index = index * shape[i] + indices[i];
    }
    return data[index];
  }

  const Shape &GetShape() const { return shape; }

private:
  std::vector<T> data;
  Shape shape;
  const std::string kDataType = typeid(T).name();
};
} // namespace llama2
