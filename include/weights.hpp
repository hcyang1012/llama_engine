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
  TransformerWeights(const Config &config, const std::string &chkpt_file)
      : config(config) {
    load_weights(chkpt_file);
  }

  std::shared_ptr<Tensor<T>> TokenEmbeddingTable() const {
    return token_embedding_table;
  }
  std::shared_ptr<Tensor<T>> RMSAttnWeight() const { return rms_attn_weight; }
  std::shared_ptr<Tensor<T>> RMSFFNWeight() const { return rms_ffn_weight; }
  std::shared_ptr<Tensor<T>> WQ() const { return wq; }
  std::shared_ptr<Tensor<T>> WK() const { return wk; }
  std::shared_ptr<Tensor<T>> WV() const { return wv; }
  std::shared_ptr<Tensor<T>> WO() const { return wo; }
  std::shared_ptr<Tensor<T>> WFFN1() const { return wffn1; }
  std::shared_ptr<Tensor<T>> WFFN2() const { return wffn2; }
  std::shared_ptr<Tensor<T>> WFFN3() const { return wffn3; }
  std::shared_ptr<Tensor<T>> RMSFinalWeight() const { return rms_final_weight; }
  std::shared_ptr<Tensor<T>> WCLS() const { return wcls; }

private:
  void load_weights(const std::string &chkpt_file) {
    // Load weights from the checkpoint file
  }
  const Config &config;
  std::shared_ptr<Tensor<T>>
      token_embedding_table; ///< Token embedding table. Shape: [vocab_size,
                             ///< dim]
  std::shared_ptr<Tensor<T>>
      rms_attn_weight; ///< RMS attention weight. Shape: [layer, dim]

  std::shared_ptr<Tensor<T>> rms_ffn_weight; ///< RMS FFN weight. Shape:
                                             ///< [layer, dim]
  std::shared_ptr<Tensor<T>> wq; ///< Query weight. Shape: [layer, dim]
  std::shared_ptr<Tensor<T>> wk; ///< Key weight. Shape: [layer, dim]
  std::shared_ptr<Tensor<T>> wv; ///< Value weight. Shape: [layer, dim]
  std::shared_ptr<Tensor<T>> wo; ///< Output weight. Shape: [layer, dim]
  std::shared_ptr<Tensor<T>>
      wffn1; ///< FFN weight 1. Shape: [layer, hidden_dim, dim]
  std::shared_ptr<Tensor<T>>
      wffn2; ///< FFN weight 2. Shape: [layer, dim, hidden_dim]
  std::shared_ptr<Tensor<T>>
      wffn3; ///< FFN weight 3. Shape: [layer, hidden_dim, dim]

  std::shared_ptr<Tensor<T>> rms_final_weight; ///< RMS final weight. Shape:
                                               ///< [dim, vocab_size]
  std::shared_ptr<Tensor<T>> wcls; ///< Classification weight for the logit.
                                   ///< Shape: [dim, vocab_size]
};

} // namespace llama2
