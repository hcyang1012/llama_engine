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

namespace llama2 {

template <typename T>
class TransformerWeights {
 public:
  TransformerWeights(const Config &config, const T *p_weights)
      : config(config) {
    load_weights(p_weights, config.VocabSize() > 0);
  }

  const T *TokenEmbeddingTable() const { return token_embedding_table_; }
  const T *RMSAttnWeight() const { return rms_attn_weight_; }
  const T *RMSFFNWeight() const { return rms_ffn_weight_; }
  const T *WQ() const { return wq_; }
  const T *WK() const { return wk_; }
  const T *WV() const { return wv_; }
  const T *WO() const { return wo_; }
  const T *W1() const { return w1_; }
  const T *W2() const { return w2_; }
  const T *W3() const { return w3_; }
  const T *RMSFinalWeight() const { return rms_final_weight_; }
  const T *WCLS() const { return wcls_; }

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
};

}  // namespace llama2
