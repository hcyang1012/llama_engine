/**
 * @file tokenizer.hpp
 * @brief Header for the Tokenizer class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-07
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <string>
#include <vector>

// Project Headers

// Third-party Headers

namespace llama2 {
template <typename T> class Tokenizer {
public:
  struct TokenIndex {
    std::string str;
    size_t id;
  };

  Tokenizer(const std::string &vocab_file, const size_t vocab_size)
      : vocab_size_(vocab_size) {
    load_vocab(vocab_file);
  }

private:
  std::vector<std::string> vocab_;
  std::vector<T> vocab_scores_;
  std::vector<TokenIndex> sorted_vocab_;
  size_t vocab_size_;
  uint32_t max_token_length_;
  std::vector<std::string> pices_;

  void load_vocab(const std::string &vocab_file) {
    vocab_.resize(vocab_size_);
    vocab_scores_.resize(vocab_size_);

    // Initize pieces
    for (int c = 0; c < 128; c++) {
      pices_.push_back(std::string(1, c));
    }

    // Load the vocabulary file
    std::ifstream if_vocab_file(vocab_file, std::ios::binary);
    if (!if_vocab_file.is_open()) {
      throw std::runtime_error("Failed to open the vocabulary file.");
    }

    // Load the vocabulary
    if_vocab_file.read(reinterpret_cast<char *>(&max_token_length_),
                       sizeof(max_token_length_));
    for (size_t i = 0; i < vocab_size_; i++) {
      int32_t len;
      float score;
      if_vocab_file.read(reinterpret_cast<char *>(&score), sizeof(score));
      vocab_scores_[i] = score;
      if_vocab_file.read(reinterpret_cast<char *>(&len), sizeof(len));
      vocab_[i].resize(len + 1);
      if_vocab_file.read(&vocab_[i][0], len);
      vocab_[i][len] = '\0';
    }
  }
};
} // namespace llama2
