/**
 * @file tokenizer.hpp
 * @brief Header for the Tokenizer class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-07
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Project Headers

// Third-party Headers

namespace llama {

enum SpecialTokens : int {
  UNK_00 = 0,
  BOS_01 = 1,
  EOS_02 = 2,
  NUM_SPECIAL_TOKENS = 3
};

template <typename T>
class Tokenizer {
 public:
  Tokenizer(const std::string &vocab_file, const size_t vocab_size)
      : vocab_size_(vocab_size) {
    load_vocab(vocab_file);
  }

  const size_t &VocabSize() const { return vocab_size_; }
  const auto &Vocab() const { return vocab_; }
  const auto &VocabScores() const { return vocab_scores_; }
  const auto &BytePieces() const { return pices_; }
  const auto &MaxTokenLength() const { return max_token_length_; }
  const auto &VocabMap() const { return vocab_map_; }

 private:
  std::vector<std::string> vocab_;
  std::vector<T> vocab_scores_;
  size_t vocab_size_;
  uint32_t max_token_length_;
  std::vector<std::string> pices_;
  std::map<std::string, size_t> vocab_map_;

  void load_vocab(const std::string &vocab_file) {
    // Initize pieces
    for (size_t i = 0; i < 256; i++) {
      std::stringstream ss;
      ss << static_cast<char>(i);
      pices_.push_back(ss.str());
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
      vocab_scores_.push_back(score);
      if_vocab_file.read(reinterpret_cast<char *>(&len), sizeof(len));
      std::string new_vocab;
      new_vocab.resize(len);
      if_vocab_file.read(&new_vocab[0], len);
      vocab_.push_back(new_vocab);
      vocab_map_[new_vocab] = i;
    }
  }
};
}  // namespace llama
