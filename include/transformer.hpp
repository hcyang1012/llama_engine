/**
 * @file transformer.hpp
 * @brief Header for the Transformer class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// C++ System-Headers
#include <cstddef>
#include <memory>

// Project Headers
#include <config.hpp>
#include <run_state.hpp>
#include <sampler.hpp>
#include <tensor.hpp>
#include <tokenizer.hpp>
#include <weights.hpp>

// Third-party Headers

namespace llama2 {

template <typename T>
class Transformer {
 public:
  Transformer(const std::string &ckpt_file) { load_checkpoint(ckpt_file); }
  ~Transformer() {}

  const auto &GetConfig() const { return *config_; }
  const auto &GetWeights() const { return *weights_; }
  const auto &GetRunState() const { return *run_state_; }

  auto &GetRunState() { return *run_state_; }

  void Generate(const Tokenizer<T> &tokenizer, const Sampler &sampler,
                const std::string &prompt, const size_t steps) {}

 private:
  void load_checkpoint(const std::string &ckpt_file) {
    // Load the configuration file
    std::ifstream if_chkpt_file(ckpt_file, std::ios::binary);
    if (!if_chkpt_file.is_open()) {
      throw std::runtime_error("Failed to open the checkpoint file.");
    }

    // Load the configuration
    config_ = std::make_unique<Config>(if_chkpt_file);

    // Load the weights
    if_chkpt_file.seekg(0, std::ios::end);
    file_size_ = if_chkpt_file.tellg();
    if_chkpt_file.close();

    fd_ = open(ckpt_file.c_str(), O_RDONLY);
    if (fd_ == -1) {
      throw std::runtime_error("Failed to open the checkpoint file.");
    }

    mapped_file_ = static_cast<T *>(
        mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));

    if (mapped_file_ == MAP_FAILED) {
      throw std::runtime_error("Failed to map the checkpoint file.");
    }

    T *weights_ptr = mapped_file_ + Config::Size() / sizeof(T);

    load_weights(weights_ptr);
    run_state_ = std::make_unique<RunState<T>>(*config_);
  }

  void load_weights(T *weights_ptr) {
    weights_ = std::make_unique<TransformerWeights<T>>(*config_, weights_ptr);
  }

  std::unique_ptr<Config> config_;  ///< Hyperparameters of the Transformer
  std::unique_ptr<TransformerWeights<T>>
      weights_;                             ///< Weights of the Transformer
  std::unique_ptr<RunState<T>> run_state_;  ///< Run state of the Transformer

  int fd_;             // file descriptor for the memory mapped file
  ssize_t file_size_;  // size of the memory mapped file
  T *mapped_file_;     // pointer to the memory mapped file
};

}  // namespace llama2
