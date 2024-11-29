/**
 * @file decoder.hpp
 * @brief Header file for the Decoder class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-25
 */

#pragma once

// C System-Headers
#include <cstdio>

// C++ System-Headers
#include <sstream>
#include <string>
// Project Headers
#include <tokenizer.hpp>

// Third-party Headers

namespace llama {
template <typename T>
class Decoder {
 public:
  static std::string Decode(const Tokenizer<T>& tokenizer, const int prev_token,
                            const int token,
                            const SpecialTokens& special_tokens) {
    std::stringstream ss;

    auto str_buffer = tokenizer.Vocab()[token];
    const char* piece = str_buffer.data();

    // following BOS (1) token, sentencepiece decoder strips any leading
    // whitespace
    if (prev_token == special_tokens.GetToken(SpecialTokens::Idx::IDX_BOS_01) &&
        piece[0] == ' ') {
      piece++;
    }

    unsigned char byte = 0;
    if (sscanf(piece, "<0x%02hhX>", &byte) == 1) {
      ss << tokenizer.BytePieces()[byte];
    } else {
      ss << piece;
    }
    return ss.str();
  }

 private:
};
}  // namespace llama
