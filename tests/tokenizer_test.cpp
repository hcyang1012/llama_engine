#include "tokenizer.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>

#include "reference.hpp"
#include "transformer.hpp"

class TokenizerTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    reference::build_transformer(&ref_transformer_, kChkPointPath.c_str());
    reference::build_tokenizer(&ref_tokenizer_, kTokenizerBinPath.c_str(),
                               ref_transformer_.config.vocab_size);

    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
    tokenizer_ = std::make_unique<llama2::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  reference::Transformer ref_transformer_;
  reference::Tokenizer ref_tokenizer_;

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::Tokenizer<float>> tokenizer_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";
};

TEST_F(TokenizerTest, VocabSizeTest) {
  EXPECT_EQ(tokenizer_->VocabSize(), ref_tokenizer_.vocab_size);
  EXPECT_EQ(tokenizer_->Vocab().size(), ref_tokenizer_.vocab_size);
  EXPECT_EQ(tokenizer_->VocabScores().size(), ref_tokenizer_.vocab_size);
}

TEST_F(TokenizerTest, BytePicesTest) {
  const auto& kBytePieces = tokenizer_->BytePieces();
  const auto& refBytePieces = ref_tokenizer_.byte_pieces;
  for (size_t i = 0; i < 256; i++) {
    EXPECT_EQ(static_cast<unsigned char>(kBytePieces[i][0]),
              refBytePieces[i * 2]);
    EXPECT_EQ(static_cast<unsigned char>(kBytePieces[i][1]),
              refBytePieces[i * 2 + 1]);
  }
}

TEST_F(TokenizerTest, MaxTokenLengthTest) {
  EXPECT_EQ(tokenizer_->MaxTokenLength(), ref_tokenizer_.max_token_length);
}

TEST_F(TokenizerTest, VocabTest) {
  const auto& kVocab = tokenizer_->Vocab();
  const auto& refVocab = ref_tokenizer_.vocab;
  for (size_t i = 0; i < tokenizer_->VocabSize(); i++) {
    EXPECT_EQ(kVocab[i], refVocab[i]);
  }
}