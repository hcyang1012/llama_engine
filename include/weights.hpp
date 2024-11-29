/**
 * @file weights.hpp
 * @brief Header file for the Weights class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
#include <memory>
#include <string>
// Project Headers
#include <config.hpp>
#include <tensor.hpp>

// Third-party Headers

namespace llama {

template <typename T>
class TransformerWeights {
 public:
  TransformerWeights(const std::string &checkpoint_path, const Config &config,
                     const DeviceType device_type)
      : config(config), device_type_(device_type) {
    load_weights(checkpoint_path);
  }

  ~TransformerWeights() {
    if (mapped_file_ != nullptr) {
      munmap(mapped_file_, file_size_);
    }
    if (fd_ != -1) {
      close(fd_);
    }
  }

  const T *TokenEmbeddingTable() const { return token_embedding_table_; }
  const T *RMSAttnWeight() const { return rms_attn_weight_; }
  const Tensor<T> RMSAttnWeight(const size_t layer) const {
    return Tensor<T>(rms_attn_weight_ + layer * config.Dim(),
                     {static_cast<size_t>(config.Dim())}, device_type_);
  }
  const T *RMSFFNWeight() const { return rms_ffn_weight_; }
  const Tensor<T> RMSFFNWeight(const size_t layer) const {
    return Tensor<T>(rms_ffn_weight_ + layer * config.Dim(),
                     {static_cast<size_t>(config.Dim())}, device_type_);
  }

  const T *WQ() const { return wq_; }
  const Tensor<T> WQ(const size_t layer) const {
    return Tensor<T>(
        wq_ + layer * config.Dim() * config.NumHeads() * config.HeadDim(),
        {static_cast<size_t>(config.HeadDim()),
         static_cast<size_t>(config.NumHeads()),
         static_cast<size_t>(config.Dim())},
        device_type_);
  }
  const T *WK() const { return wk_; }
  const Tensor<T> WK(const size_t layer) const {
    return Tensor<T>(
        wk_ + layer * config.Dim() * config.NumKVHeads() * config.HeadDim(),
        {static_cast<size_t>(config.HeadDim()),
         static_cast<size_t>(config.NumKVHeads()),
         static_cast<size_t>(config.Dim())},
        device_type_);
  }

  const T *WV() const { return wv_; }
  const Tensor<T> WV(const size_t layer) const {
    return Tensor<T>(
        wv_ + layer * config.Dim() * config.NumKVHeads() * config.HeadDim(),
        {static_cast<size_t>(config.HeadDim()),
         static_cast<size_t>(config.NumKVHeads()),
         static_cast<size_t>(config.Dim())},
        device_type_);
  }

  const T *WO() const { return wo_; }
  const Tensor<T> WO(const size_t layer) const {
    return Tensor<T>(
        wo_ + layer * config.NumHeads() * config.HeadDim() * config.Dim(),
        {static_cast<size_t>(config.Dim()),
         static_cast<size_t>(config.HeadDim()),
         static_cast<size_t>(config.NumHeads())},
        device_type_);
  }

  const T *W1() const { return w1_; }
  const Tensor<T> W1(const size_t layer) const {
    return Tensor<T>(w1_ + layer * config.HiddenDim() * config.Dim(),
                     {static_cast<size_t>(config.Dim()),
                      static_cast<size_t>(config.HiddenDim())},
                     device_type_);
  }
  const T *W2() const { return w2_; }
  const Tensor<T> W2(const size_t layer) const {
    return Tensor<T>(w2_ + layer * config.Dim() * config.HiddenDim(),
                     {static_cast<size_t>(config.HiddenDim()),
                      static_cast<size_t>(config.Dim())},
                     device_type_);
  }
  const T *W3() const { return w3_; }
  const Tensor<T> W3(const size_t layer) const {
    return Tensor<T>(w3_ + layer * config.HiddenDim() * config.Dim(),
                     {static_cast<size_t>(config.Dim()),
                      static_cast<size_t>(config.HiddenDim())},
                     device_type_);
  }
  const Tensor<T> RMSFinalWeight() const {
    return Tensor<T>(rms_final_weight_, {static_cast<size_t>(config.Dim())},
                     device_type_);
  }
  const Tensor<T> WCLS() const {
    return Tensor<T>(wcls_,
                     {static_cast<size_t>(config.Dim()),
                      static_cast<size_t>(config.VocabSize())},
                     device_type_);
  }

 private:
  void load_weights(const T *p_weights, const bool shared_weights) {
    // Load weights from the checkpoint file
    const int kHeadSize = config.Dim() / config.NumHeads();

    const uint64_t kNumLayers = static_cast<uint64_t>(config.NumLayers());

    const size_t kTokenEmbeddingTableSize =
        static_cast<size_t>(config.VocabSize()) * config.Dim();
    token_embedding_table_ = p_weights;
    p_weights += kTokenEmbeddingTableSize;

    const size_t kRmsAttnWeightSize = kNumLayers * config.Dim();
    rms_attn_weight_ = p_weights;
    p_weights += kRmsAttnWeightSize;

    const size_t kWQSize =
        kNumLayers * config.Dim() * (config.NumHeads() * kHeadSize);
    wq_ = p_weights;
    p_weights += kWQSize;

    const size_t kWKSize =
        kNumLayers * config.Dim() * (config.NumKVHeads() * kHeadSize);
    wk_ = p_weights;
    p_weights += kWKSize;

    const size_t kWVSize =
        kNumLayers * config.Dim() * (config.NumKVHeads() * kHeadSize);
    wv_ = p_weights;
    p_weights += kWVSize;

    const size_t kWOSize =
        kNumLayers * config.Dim() * (config.NumHeads() * kHeadSize);
    wo_ = p_weights;
    p_weights += kWOSize;

    const size_t kRmsFFNWeightSize = kNumLayers * config.Dim();
    rms_ffn_weight_ = p_weights;
    p_weights += kRmsFFNWeightSize;

    const size_t kW1Size = kNumLayers * config.HiddenDim() * config.Dim();
    w1_ = p_weights;
    p_weights += kW1Size;

    const size_t kW2Size = kNumLayers * config.Dim() * config.HiddenDim();
    w2_ = p_weights;
    p_weights += kW2Size;

    const size_t kW3Size = kNumLayers * config.HiddenDim() * config.Dim();
    w3_ = p_weights;
    p_weights += kW3Size;

    const size_t kRmsFinalWeightSize = config.Dim();
    rms_final_weight_ = p_weights;
    p_weights += kRmsFinalWeightSize;

    p_weights += config.SeqLen() * kHeadSize /
                 2;  // Skip what used to be freq_cis_real (for RoPE)
    p_weights += config.SeqLen() * kHeadSize /
                 2;  // Skip what used to be freq_cis_imag (for RoPE)

    wcls_ = shared_weights ? token_embedding_table_ : p_weights;
  }

  void load_weights(const std::string &ckpt_file) {
    // Load the configuration file
    std::ifstream if_chkpt_file(ckpt_file, std::ios::binary);
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

    load_weights(weights_ptr, config.VocabSize() > 0);
  }

  const Config &config;
  const T *token_embedding_table_;  ///< Token embedding table. Shape:
                                    ///< [vocab_size, dim]
  const T *rms_attn_weight_;  ///< RMS attention weight. Shape: [layer, dim]

  const T *rms_ffn_weight_;  ///< RMS FFN weight. Shape:
                             ///< [layer, dim]
  const T *wq_;              ///< Query weight. Shape: [layer, dim]
  const T *wk_;              ///< Key weight. Shape: [layer, dim]
  const T *wv_;              ///< Value weight. Shape: [layer, dim]
  const T *wo_;              ///< Output weight. Shape: [layer, dim]
  const T *w1_;              ///< FFN weight 1. Shape: [layer, hidden_dim, dim]
  const T *w2_;              ///< FFN weight 2. Shape: [layer, dim, hidden_dim]
  const T *w3_;              ///< FFN weight 3. Shape: [layer, hidden_dim, dim]

  const T *rms_final_weight_;  ///< RMS final weight. Shape:
                               ///< [dim, vocab_size]
  const T *wcls_;              ///< Classification weight for the logit.
  ///< Shape: [dim, vocab_size]

  const DeviceType device_type_;

  int fd_;             // file descriptor for the memory mapped file
  ssize_t file_size_;  // size of the memory mapped file
  T *mapped_file_;     // pointer to the memory mapped file
};

}  // namespace llama
