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
#include <string>
// Project Headers

// Third-party Headers

namespace llama2 {

/**
 * @brief Configuration class for the Transformer model.
 */
class Config {
public:
  Config(const std::string &config_file);

  size_t Dim() const;        ///< Get the dimension of the Transformer
  size_t HiddenDim() const;  ///< Get the dimension of the FeedForward hidden
                             ///< layer (FFN) in the Transformer
  size_t NumLayers() const;  ///< Get the number of layers in the Transformer
  size_t NumHeads() const;   ///< Get the number of attention heads in the
                             ///< Transformer
  size_t NumKVHeads() const; ///< Get the number of attention heads for key and
                             ///< value in the Transformer
  size_t VocabSize() const;  ///< Get the size of the vocabulary
  size_t SeqLen() const;     ///< Get the maximum sequence length

private:
  void load_config(const std::string &config_file);
  size_t kDim;        ///< Transformer dimension
  size_t kHiddenDim;  ///< Dimension of the FeedForward hidden layer (FFN)
                      ///< in the Transformer
  size_t kNumLayers;  ///< Number of layers in the Transformer
  size_t kNumHeads;   ///< Number of attention heads in the Transformer
  size_t kNumKVHeads; ///< Number of attention heads for key and value in
                      ///< the Transformer (Can be less than kNumHeads
                      ///< because of multi-query attention)
  size_t kVocabSize;  ///< Size of the vocabulary
  size_t kSeqLen;     ///< Maximum sequence length
};
} // namespace llama2