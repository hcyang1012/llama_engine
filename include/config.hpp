/**
 * @file config.hpp
 * @brief Header file for the TransformerConfig class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstddef>
#include <fstream>
#include <string>
// Project Headers
#include <dtypes.hpp>
// Third-party Headers

namespace llama {

/**
 * @brief Configuration class for the Transformer model.
 */
class TransformerConfig {
 public:
  TransformerConfig(const std::string &config_file) {
    std::ifstream if_config_file(config_file, std::ios::binary);
    load_config(if_config_file);
  }
  llama_uint32_t Dim() const;        ///< Get the dimension of the Transformer
  llama_uint32_t HiddenDim() const;  ///< Get the dimension of the FeedForward
                                     ///< hidden layer (FFN) in the Transformer
  llama_uint32_t NumLayers()
      const;  ///< Get the number of layers in the Transformer
  llama_uint32_t NumHeads() const;    ///< Get the number of attention heads in
                                      ///< the Transformer
  llama_uint32_t NumKVHeads() const;  ///< Get the number of attention heads for
                                      ///< key and value in the Transformer
  llama_uint32_t HeadDim() const;  ///< Get the dimension of each attention head
                                   ///< in the Transformer (Dim / NumHeads)
  llama_uint32_t KVHeadDim()
      const;  ///< Get the dimension of each key and value head
              ///< in the Transformer (Dim / NumKVHeads)
  llama_uint32_t KVMul() const;
  llama_uint32_t VocabSize() const;  ///< Get the size of the vocabulary
  llama_uint32_t SeqLen() const;     ///< Get the maximum sequence length

  bool ShareWeights() const;  ///< Get whether to share weights between layers

  virtual llama_float Freq() const = 0;

  static size_t Size();  ///< Get the size of the configuration in bytes

 private:
  llama_uint32_t kDim;         ///< Transformer dimension
  llama_uint32_t kHiddenDim;   ///< Dimension of the FeedForward hidden layer
                               ///< (FFN) in the Transformer
  llama_uint32_t kNumLayers;   ///< Number of layers in the Transformer
  llama_uint32_t kNumHeads;    ///< Number of attention heads in the Transformer
  llama_uint32_t kNumKVHeads;  ///< Number of attention heads for key and value
                               ///< in the Transformer (Can be less than
                               ///< kNumHeads because of multi-query attention)
  llama_uint32_t kHeadDim;     ///< Dimension of each attention head in the
                               ///< Transformer (kDim / kNumHeads)
  llama_uint32_t kKVHeadDim;   ///< Dimension of each key and value head in the
                               ///< Transformer (kDim * kNumKVHeads / kNumHeads)
  llama_uint32_t
      kKVMul;  /// Number of heads for key and value per head for query
  llama_uint32_t kVocabSize;  ///< Size of the vocabulary
  llama_uint32_t kSeqLen;     ///< Maximum sequence length
  bool kShareWeights;         ///< Whether to share weights between layers

  void load_config(std::ifstream &config_file);
};

llama_uint32_t TransformerConfig::Dim() const { return kDim; }

llama_uint32_t TransformerConfig::HiddenDim() const { return kHiddenDim; }

llama_uint32_t TransformerConfig::NumLayers() const { return kNumLayers; }

llama_uint32_t TransformerConfig::NumHeads() const { return kNumHeads; }

llama_uint32_t TransformerConfig::NumKVHeads() const { return kNumKVHeads; }

llama_uint32_t TransformerConfig::HeadDim() const { return kHeadDim; }

llama_uint32_t TransformerConfig::KVHeadDim() const { return kKVHeadDim; }

llama_uint32_t TransformerConfig::KVMul() const { return kKVMul; }

llama_uint32_t TransformerConfig::VocabSize() const { return kVocabSize; }

llama_uint32_t TransformerConfig::SeqLen() const { return kSeqLen; }

bool TransformerConfig::ShareWeights() const { return kShareWeights; }

void TransformerConfig::load_config(std::ifstream &config_file) {
  config_file.read(reinterpret_cast<char *>(&kDim), sizeof(kDim));
  config_file.read(reinterpret_cast<char *>(&kHiddenDim), sizeof(kHiddenDim));
  config_file.read(reinterpret_cast<char *>(&kNumLayers), sizeof(kNumLayers));
  config_file.read(reinterpret_cast<char *>(&kNumHeads), sizeof(kNumHeads));
  config_file.read(reinterpret_cast<char *>(&kNumKVHeads), sizeof(kNumKVHeads));

  int32_t vocab_size;
  config_file.read(reinterpret_cast<char *>(&vocab_size), sizeof(vocab_size));
  kVocabSize = static_cast<llama_uint32_t>(std::abs(vocab_size));
  kShareWeights = vocab_size > 0;
  config_file.read(reinterpret_cast<char *>(&kSeqLen), sizeof(kSeqLen));

  kKVHeadDim = Dim() * NumKVHeads() / NumHeads();
  kHeadDim = Dim() / NumHeads();
  kKVMul = NumHeads() / NumKVHeads();
}

size_t TransformerConfig::Size() {
  return sizeof(kDim) + sizeof(kHiddenDim) + sizeof(kNumLayers) +
         sizeof(kNumHeads) + sizeof(kNumKVHeads) + sizeof(kVocabSize) +
         sizeof(kSeqLen);
}

class ConfigLlama2 : public TransformerConfig {
 public:
  ConfigLlama2(const std::string &config_file)
      : TransformerConfig(config_file) {}
  llama_float Freq() const override { return 10000.0f; }
};

class ConfigLlama3 : public TransformerConfig {
 public:
  ConfigLlama3(const std::string &config_file)
      : TransformerConfig(config_file) {}
  llama_float Freq() const override { return 500000.0f; }
};

}  // namespace llama