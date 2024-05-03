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

template <typename T> class TransformerWeights {
public:
  TransformerWeights(const Config &config, const T *p_weights)
      : config(config) {
    load_weights(p_weights, config.VocabSize() > 0);
  }

  const T *TokenEmbeddingTable() const { return token_embedding_table; }
  const T *RMSAttnWeight() const { return rms_attn_weight; }
  const T *RMSFFNWeight() const { return rms_ffn_weight; }
  const T *WQ() const { return wq; }
  const T *WK() const { return wk; }
  const T *WV() const { return wv; }
  const T *WO() const { return wo; }
  const T *WFFN1() const { return wffn1; }
  const T *WFFN2() const { return wffn2; }
  const T *WFFN3() const { return wffn3; }
  const T *RMSFinalWeight() const { return rms_final_weight; }
  const T *WCLS() const { return wcls; }

private:
  void load_weights(const T *p_weights, const bool shared_weights) {
    // Load weights from the checkpoint file
    const int kHeadSize = config.Dim() / config.NumHeads();

    const uint64_t kNumLayers = static_cast<uint64_t>(config.NumLayers());

    const size_t kTokenEmbeddingTableSize =
        static_cast<size_t>(config.VocabSize()) * config.Dim();
    token_embedding_table = p_weights;
    p_weights += kTokenEmbeddingTableSize;

    const size_t kRmsAttnWeightSize = kNumLayers * config.Dim();
    rms_attn_weight = p_weights;
    p_weights += kRmsAttnWeightSize;

    const size_t kWQSize =
        kNumLayers * config.Dim() * (config.NumHeads() * kHeadSize);
    wq = p_weights;
    p_weights += kWQSize;

    const size_t kWKSize =
        kNumLayers * config.Dim() * (config.NumKVHeads() * kHeadSize);
    wk = p_weights;
    p_weights += kWKSize;

    const size_t kWVSize =
        kNumLayers * config.Dim() * (config.NumKVHeads() * kHeadSize);
    wv = p_weights;
    p_weights += kWVSize;

    const size_t kWOSize =
        kNumLayers * config.Dim() * (config.NumHeads() * kHeadSize);
    wo = p_weights;
    p_weights += kWOSize;

    const size_t kRmsFFNWeightSize = kNumLayers * config.Dim();
    rms_ffn_weight = p_weights;
    p_weights += kRmsFFNWeightSize;

    const size_t kWFFN1Size = kNumLayers * config.HiddenDim() * config.Dim();
    wffn1 = p_weights;
    p_weights += kWFFN1Size;

    const size_t kWFFN2Size = kNumLayers * config.Dim() * config.HiddenDim();
    wffn2 = p_weights;
    p_weights += kWFFN2Size;

    const size_t kWFFN3Size = kNumLayers * config.HiddenDim() * config.Dim();
    wffn3 = p_weights;
    p_weights += kWFFN3Size;

    const size_t kRmsFinalWeightSize = config.Dim();
    rms_final_weight = p_weights;
    p_weights += kRmsFinalWeightSize;

    p_weights += config.SeqLen() * kHeadSize /
                 2; // Skip what used to be freq_cis_real (for RoPE)
    p_weights += config.SeqLen() * kHeadSize /
                 2; // Skip what used to be freq_cis_imag (for RoPE)

    wcls = shared_weights ? token_embedding_table : p_weights;
  }
  const Config &config;
  const T *token_embedding_table; ///< Token embedding table. Shape:
                                  ///< [vocab_size, dim]
  const T *rms_attn_weight;       ///< RMS attention weight. Shape: [layer, dim]

  const T *rms_ffn_weight; ///< RMS FFN weight. Shape:
                           ///< [layer, dim]
  const T *wq;             ///< Query weight. Shape: [layer, dim]
  const T *wk;             ///< Key weight. Shape: [layer, dim]
  const T *wv;             ///< Value weight. Shape: [layer, dim]
  const T *wo;             ///< Output weight. Shape: [layer, dim]
  const T *wffn1;          ///< FFN weight 1. Shape: [layer, hidden_dim, dim]
  const T *wffn2;          ///< FFN weight 2. Shape: [layer, dim, hidden_dim]
  const T *wffn3;          ///< FFN weight 3. Shape: [layer, hidden_dim, dim]

  const T *rms_final_weight; ///< RMS final weight. Shape:
                             ///< [dim, vocab_size]
  const T *wcls;             ///< Classification weight for the logit.
                             ///< Shape: [dim, vocab_size]
};

} // namespace llama2
