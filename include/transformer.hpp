/**
 * @file transformer.hpp
 * @brief Header for the Transformer class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers
#include <unistd.h>

// C++ System-Headers
#include <cstddef>
#include <memory>

// Project Headers
#include <config.hpp>
#include <run_state.hpp>
#include <tensor.hpp>
#include <weights.hpp>

// Third-party Headers

namespace llama2 {

template <typename T> class Transformer {
public:
  Transformer(const std::string &config_file, const std::string &chkpt_file)
      : config_(std::make_unique<Config>(config_file)),
        run_state_(std::make_unique<RunState<T>>()),
        weights_(
            std::make_unique<TransformerWeights<T>>(*config_, chkpt_file)) {}
  ~Transformer() {}

private:
  std::unique_ptr<Config> config_;
  std::unique_ptr<RunState<T>> run_state_;
  std::unique_ptr<TransformerWeights<T>> weights_;

  int fd_;            // file descriptor for the memory mapped file
  ssize_t file_size_; // size of the memory mapped file
  T *mapped_file_;    // pointer to the memory mapped file
};

} // namespace llama2
