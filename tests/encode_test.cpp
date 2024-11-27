#include <glog/logging.h>
#include <gtest/gtest.h>

#include <string>

#include "encoder.hpp"
#include "references/reference_llama2.cpp"
#include "tokenizer.hpp"
#include "transformer.hpp"

class EncodeTest : public ::testing::Test {
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

TEST_F(EncodeTest, SampleText) {
  const std::string kPrompt = "Who are you?";

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  const size_t kPromptSize = kPrompt.length() + 3;  // +3 for '\0', ?BOS, ?EOS
  int *prompt_tokens = (int *)malloc(kPromptSize * sizeof(int));
  reference::encode(&ref_tokenizer_, kPrompt.c_str(), 1, 0, prompt_tokens,
                    &num_prompt_tokens);
  EXPECT_GE(num_prompt_tokens, 1);
  auto encoder = llama2::Encoder<float>(*tokenizer_, kPrompt, true, false);
  auto result = encoder.PromptTokens();
  EXPECT_TRUE(std::equal(prompt_tokens, prompt_tokens + num_prompt_tokens,
                         result.data()));

  free(prompt_tokens);
}