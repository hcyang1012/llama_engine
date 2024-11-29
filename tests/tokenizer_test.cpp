#include "tokenizer.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>

#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif
#include "transformer.hpp"

class TokenizerTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    reference_llama2::build_transformer(&ref_transformer_,
                                        kChkPointPath.c_str());
    reference_llama2::build_tokenizer(&ref_tokenizer_,
                                      kTokenizerBinPath.c_str(),
                                      ref_transformer_.config.vocab_size);

    transformer_ = std::make_unique<llama::Transformer<float>>(
        kChkPointPath, *op_set_, llama::SpecialTokensLlama2());
    tokenizer_ = std::make_unique<llama::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  reference_llama2::Transformer ref_transformer_;
  reference_llama2::Tokenizer ref_tokenizer_;

  std::unique_ptr<llama::Transformer<float>> transformer_;
  std::unique_ptr<llama::Tokenizer<float>> tokenizer_;
  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(llama::DeviceType::CPU);

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

TEST_F(TokenizerTest, VocabSortTest) {
  const auto& kSortedVocab = tokenizer_->VocabMap();
  reference_llama2::TokenIndex* ref_sorted_vocab =
      (reference_llama2::TokenIndex*)malloc(
          ref_tokenizer_.vocab_size * sizeof(reference_llama2::TokenIndex));

  for (int i = 0; i < ref_tokenizer_.vocab_size; i++) {
    ref_sorted_vocab[i].str = ref_tokenizer_.vocab[i];
    ref_sorted_vocab[i].id = i;
  }

  qsort(ref_sorted_vocab, ref_tokenizer_.vocab_size,
        sizeof(reference_llama2::TokenIndex), reference_llama2::compare_tokens);

  for (size_t i = 0; i < ref_tokenizer_.vocab_size; i++) {
    EXPECT_EQ(kSortedVocab.at(ref_sorted_vocab[i].str), ref_sorted_vocab[i].id);
  }

  free(ref_sorted_vocab);
}