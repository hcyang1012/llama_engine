/**
 * @file config.hpp
 * @brief Header file for the Config class.
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
#include <dtypes.h>
// Third-party Headers

namespace llama {

/**
 * @brief Configuration class for the Transformer model.
 */
class Config {
 public:
  Config(std::ifstream &config_file);  ///< Constructor

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

  static size_t Size();  ///< Get the size of the configuration in bytes

 private:
  void load_config(std::ifstream &config_file);
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
};

Config::Config(std::ifstream &config_file) { load_config(config_file); }

llama_uint32_t Config::Dim() const { return kDim; }

llama_uint32_t Config::HiddenDim() const { return kHiddenDim; }

llama_uint32_t Config::NumLayers() const { return kNumLayers; }

llama_uint32_t Config::NumHeads() const { return kNumHeads; }

llama_uint32_t Config::NumKVHeads() const { return kNumKVHeads; }

llama_uint32_t Config::HeadDim() const { return kHeadDim; }

llama_uint32_t Config::KVHeadDim() const { return kKVHeadDim; }

llama_uint32_t Config::KVMul() const { return kKVMul; }

llama_uint32_t Config::VocabSize() const { return kVocabSize; }

llama_uint32_t Config::SeqLen() const { return kSeqLen; }

void Config::load_config(std::ifstream &config_file) {
  config_file.read(reinterpret_cast<char *>(&kDim), sizeof(kDim));
  config_file.read(reinterpret_cast<char *>(&kHiddenDim), sizeof(kHiddenDim));
  config_file.read(reinterpret_cast<char *>(&kNumLayers), sizeof(kNumLayers));
  config_file.read(reinterpret_cast<char *>(&kNumHeads), sizeof(kNumHeads));
  config_file.read(reinterpret_cast<char *>(&kNumKVHeads), sizeof(kNumKVHeads));
  config_file.read(reinterpret_cast<char *>(&kVocabSize), sizeof(kVocabSize));
  config_file.read(reinterpret_cast<char *>(&kSeqLen), sizeof(kSeqLen));

  kVocabSize = VocabSize();
  kKVHeadDim = Dim() * NumKVHeads() / NumHeads();
  kHeadDim = Dim() / NumHeads();
  kKVMul = NumHeads() / NumKVHeads();
}

size_t Config::Size() {
  return sizeof(kDim) + sizeof(kHiddenDim) + sizeof(kNumLayers) +
         sizeof(kNumHeads) + sizeof(kNumKVHeads) + sizeof(kVocabSize) +
         sizeof(kSeqLen);
}

}  // namespace llama