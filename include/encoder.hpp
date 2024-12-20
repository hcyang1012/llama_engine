/**
 * @file encoder.hpp
 * @brief Header for the Encoder class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <list>
#include <memory>
#include <sstream>

// Project Headers
#include <tokenizer.hpp>

// Third-party Headers
#include <glog/logging.h>
namespace llama {

template <typename T>
class Encoder {
 public:
  Encoder(const Tokenizer<T>& tokenizer, const std::string prompt,
          const bool start_with_bos, const bool end_with_eos,
          const SpecialTokens& special_tokens)
      : tokenizer_(tokenizer), prompt_(prompt) {
    encode(prompt, start_with_bos, end_with_eos, special_tokens);
  }

  const std::vector<int>& PromptTokens() const { return prompt_tokens_vec_; }

 private:
  // private members

  const Tokenizer<T>& tokenizer_;
  const std::string prompt_;

  std::list<int> prompt_tokens_;
  std::vector<int> prompt_tokens_vec_;

  void encode(const std::string& prompt, const bool start_with_bos,
              const bool end_with_eos, const SpecialTokens& special_tokens) {
    size_t n_tokens = 0;

    if (start_with_bos) {
      prompt_tokens_.push_back(
          special_tokens.GetToken(SpecialTokens::IDX_BOS_01));
    }

    // Add a dummy prefix token as default in this implementation
    if (!prompt.empty()) {
      const std::string kDummyPrefixStr = " ";
      const auto kDummyPrefix = tokenizer_.VocabMap().at(kDummyPrefixStr);
      prompt_tokens_.push_back(kDummyPrefix);
    }

    process_utf8();
    merge();

    if (end_with_eos) {
      prompt_tokens_.push_back(
          special_tokens.GetToken(SpecialTokens::IDX_EOS_02));
    }

    prompt_tokens_vec_.assign(prompt_tokens_.begin(), prompt_tokens_.end());
  }

  void process_utf8() {
    // clang-format off

    // UTF8 encoding
    /*
    | First Code Point | Last Code Point | Byte 1   | Byte 2   | Byte 3   | Byte 4   |
    |------------------|-----------------|----------|----------|----------|----------|
    | U+0000           | U+007F          | 0xxxxxxx |          |          |          |
    | U+0080           | U+07FF          | 110xxxxx | 10xxxxxx |          |          |
    | U+0800           | U+0FFF          | 1110xxxx | 10xxxxxx
    | U+10000          | U+10FFFF        | 11110xxx | 10xxxxxx | 10xxxxxx | 10xxxxxx |
    */

    // clang-format on
    std::string str_buffer;
    auto is_continuation_byte = [](const char c) -> bool {
      return (c & 0xC0) == 0x80;
    };
    for (const auto& c : prompt_) {
      if (c == '\0') {
        break;
      }

      if (is_continuation_byte(c)) {
        str_buffer.clear();
      }
      str_buffer.push_back(c);

      // While the next byte is a continuation byte and the buffer is not full
      // continue appending
      const size_t kUtf8MaxBytes = 4;
      if (is_continuation_byte(c + 1) && str_buffer.length() < kUtf8MaxBytes) {
        continue;
      }

      // We have read in a full codepoint
      // If the codepoint is in the vocab, add it as a token
      // Otherwise, encode each byte as a token (byte_fallback encoding)
      try {
        const auto kId = tokenizer_.VocabMap().at(str_buffer);
        // we found this codepoint in vocab, add it as a token
        prompt_tokens_.push_back(kId);
      } catch (const std::out_of_range& e) {
        // byte_fallback encoding: just encode each byte as a token
        // As the first three tokens are reserved for special tokens, we add 3,
        // which is SpecialTokens::NUM_SPECIAL_TOKENS
        for (const auto& byte : str_buffer) {
          prompt_tokens_.push_back(byte + SpecialTokens::NUM_SPECIAL_TOKENS);
        }
      }
      str_buffer.clear();
    }
  }

  void merge() {
    constexpr float kEpsilon = -1e10;
    const size_t kNumTokens = prompt_tokens_.size();
    // Merge the best consecutive pair each iteration, according to the scores
    // in vocab_scores
    while (true) {
      float best_score = kEpsilon;

      std::list<int>::iterator best_it = prompt_tokens_.end();
      std::size_t best_id = 0;

      auto try_merge = [&](const size_t merge_count) {
        DCHECK_LE(merge_count, 3);
        DCHECK_GE(merge_count, 2);
        auto end = prompt_tokens_.end();
        for (size_t decrease = 0; decrease < (merge_count - 1); ++decrease) {
          end--;
        }

        std::string best_str;
        for (auto it = prompt_tokens_.begin(); it != end; ++it) {
          std::stringstream ss;
          auto next = it;
          for (size_t i = 0; i < merge_count; ++i) {
            ss << tokenizer_.Vocab()[*next];
            next = (++next);
          }
          try {
            const auto kId = tokenizer_.VocabMap().at(ss.str());
            if (tokenizer_.VocabScores()[kId] > best_score) {
              best_score = tokenizer_.VocabScores()[kId];
              best_it = it;
              best_id = kId;
              best_str = ss.str();
            }
          } catch (const std::out_of_range& e) {
            // Do nothing
          }
        }
      };
      size_t merged_count = 2;
      try_merge(2);
      if (best_it == prompt_tokens_.end()) {
        try_merge(3);
        if (best_it == prompt_tokens_.end()) {
          break;
        }
        merged_count = 3;
      } else {
        // We found a pair to merge
        merged_count = 2;
      }
      *best_it = best_id;
      auto next = best_it;
      next = (++next);
      for (size_t i = 0; i < (merged_count - 1); ++i) {
        next = prompt_tokens_.erase(next);
      }
    }
  }
};
}  // namespace llama
