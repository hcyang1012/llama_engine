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

class SpecialTokens {
 public:
  enum Idx : int {
    IDX_UNK_00 = 0,
    IDX_BOS_01 = 1,
    IDX_EOS_02 = 2,
    IDX_START_HEADER_ID_03 = 3,
    IDX_SYSTEM_04 = 4,
    IDX_END_HEADER_ID_05 = 5,
    IDX_NEW_LINE_X2_06 = 6,  // '\n\n'
    IDX_EOT_ID_07 = 7,
    IDX_USER_08 = 8,
    IDX_ASSISTANT_09 = 9,
    NUM_SPECIAL_TOKENS,
  };
  virtual size_t GetToken(const Idx idx) const = 0;

 private:
};

class SpecialTokensLlama2 : public SpecialTokens {
 public:
  size_t GetToken(const Idx idx) const override {
    switch (idx) {
      case Idx::IDX_UNK_00:
        return 0;
      case Idx::IDX_BOS_01:
        return 1;
      case Idx::IDX_EOS_02:
        return 2;
      default:
        throw std::runtime_error("Invalid special token index.");
    }
  }
};

class SpecialTokensLlama3 : public SpecialTokens {
 public:
  size_t GetToken(const Idx idx) const override {
    switch (idx) {
      case Idx::IDX_UNK_00:
        return 0;
      case Idx::IDX_BOS_01:
        return 128000;
      case Idx::IDX_EOS_02:
        return 128001;
      case Idx::IDX_START_HEADER_ID_03:
        return 128006;
      case Idx::IDX_SYSTEM_04:
        return 9125;
      case Idx::IDX_END_HEADER_ID_05:
        return 128007;
      case Idx::IDX_NEW_LINE_X2_06:
        return 271;
      case Idx::IDX_EOT_ID_07:
        return 128009;
      case Idx::IDX_USER_08:
        return 882;
      case Idx::IDX_ASSISTANT_09:
        return 78191;
      default:
        throw std::runtime_error("Invalid special token index.");
    }
  }
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
