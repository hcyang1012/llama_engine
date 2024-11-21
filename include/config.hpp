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

// Third-party Headers

namespace llama2 {

/**
 * @brief Configuration class for the Transformer model.
 */
class Config {
 public:
  Config(std::ifstream &config_file);  ///< Constructor

  int32_t Dim() const;         ///< Get the dimension of the Transformer
  int32_t HiddenDim() const;   ///< Get the dimension of the FeedForward hidden
                               ///< layer (FFN) in the Transformer
  int32_t NumLayers() const;   ///< Get the number of layers in the Transformer
  int32_t NumHeads() const;    ///< Get the number of attention heads in the
                               ///< Transformer
  int32_t NumKVHeads() const;  ///< Get the number of attention heads for key
                               ///< and value in the Transformer
  int32_t HeadDim() const;     ///< Get the dimension of each attention head in
                               ///< the Transformer (Dim / NumHeads)
  int32_t KVHeadDim() const;   ///< Get the dimension of each key and value head
                               ///< in the Transformer (Dim / NumKVHeads)
  int32_t KVMul() const;
  int32_t VocabSize() const;  ///< Get the size of the vocabulary
  int32_t SeqLen() const;     ///< Get the maximum sequence length

  static size_t Size();  ///< Get the size of the configuration in bytes

 private:
  void load_config(std::ifstream &config_file);
  int32_t kDim;         ///< Transformer dimension
  int32_t kHiddenDim;   ///< Dimension of the FeedForward hidden layer (FFN)
                        ///< in the Transformer
  int32_t kNumLayers;   ///< Number of layers in the Transformer
  int32_t kNumHeads;    ///< Number of attention heads in the Transformer
  int32_t kNumKVHeads;  ///< Number of attention heads for key and value in
                        ///< the Transformer (Can be less than kNumHeads
                        ///< because of multi-query attention)
  int32_t kHeadDim;     ///< Dimension of each attention head in the Transformer
                        ///< (kDim / kNumHeads)
  int32_t kKVHeadDim;   ///< Dimension of each key and value head in the
                        ///< Transformer (kDim * kNumKVHeads / kNumHeads)
  int32_t kKVMul;
  int32_t kVocabSize;  ///< Size of the vocabulary
  int32_t kSeqLen;     ///< Maximum sequence length
};
}  // namespace llama2