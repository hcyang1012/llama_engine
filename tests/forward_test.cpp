#include <gtest/gtest.h>

#include "encoder.hpp"
#include "op.hpp"
#include "references/reference_llama2.cpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class ForwardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
    tokenizer_ = std::make_unique<llama2::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::Tokenizer<float>> tokenizer_;
};

TEST_F(ForwardTest, Test) {
  const std::string kPrompt = "Who are you?";
  // Reference Forward Stage
  float *ref_logits = nullptr;
  {
    reference::Transformer ref_transformer;
    reference::build_transformer(&ref_transformer, kChkPointPath.c_str());

    reference::Tokenizer ref_tokenizer;
    reference::build_tokenizer(&ref_tokenizer, kTokenizerBinPath.c_str(),
                               ref_transformer.config.vocab_size);

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(kPrompt.c_str()) + 3) *
                                       sizeof(int));  // +3 for '\0', ?BOS, ?EOS
    reference::encode(&ref_tokenizer, kPrompt.c_str(), 1, 0, prompt_tokens,
                      &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
      fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
      exit(EXIT_FAILURE);
    }

    // start the main loop
    long start =
        0;     // used to time our code, only initialized after first iteration
    int next;  // will store the next token in the sequence
    int token =
        prompt_tokens[0];  // kick off with the first token in the prompt
    int pos = 0;           // position in the sequence
    ref_logits = reference::forward(&ref_transformer, token, pos);
  }

  // Forward Stage
  {
    auto encoder = llama2::Encoder<float>(*tokenizer_, kPrompt, true, false);
    auto result = encoder.PromptTokens();
    auto logits = transformer_->Forward(result[0], 0);

    // Float comparison of logits
    // for (size_t i = 0; i < logits.GetShape().GetSize(); i++) {
    //   EXPECT_NEAR(logits.GetData()[i], ref_logits[i], 1e-4);
    // }
    EXPECT_TRUE(std::equal(logits.GetData(),
                           logits.GetData() + logits.GetShape().GetSize(),
                           ref_logits));
  }
}