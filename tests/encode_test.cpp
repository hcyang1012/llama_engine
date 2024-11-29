#include <glog/logging.h>
#include <gtest/gtest.h>

#include <llama.hpp>
#include <string>

#include "references/reference_llama2.cpp"

class EncodeTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    reference_llama2::build_transformer(&ref_transformer_,
                                        kChkPointPath.c_str());
    reference_llama2::build_tokenizer(&ref_tokenizer_,
                                      kTokenizerBinPath.c_str(),
                                      ref_transformer_.config.vocab_size);
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  reference_llama2::Transformer ref_transformer_;
  reference_llama2::Tokenizer ref_tokenizer_;

  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(llama::DeviceType::CPU);

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";
};

TEST_F(EncodeTest, SampleText) {
  const llama::LlamaConfig llama_config = {kChkPointPath, kTokenizerBinPath,
                                           llama::DeviceType::CPU};
  llama::Llama2<float> llama2(llama_config);

  const auto &tokenizer = llama2.GetTokenizer();
  const std::string kPrompt = "Who are you?";

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  const size_t kPromptSize = kPrompt.length() + 3;  // +3 for '\0', ?BOS, ?EOS
  int *prompt_tokens = (int *)malloc(kPromptSize * sizeof(int));
  reference_llama2::encode(&ref_tokenizer_, kPrompt.c_str(), 1, 0,
                           prompt_tokens, &num_prompt_tokens);
  EXPECT_GE(num_prompt_tokens, 1);
  auto encoder = llama::Encoder<float>(tokenizer, kPrompt, true, false,
                                       llama::SpecialTokensLlama2());
  auto result = encoder.PromptTokens();
  EXPECT_TRUE(std::equal(prompt_tokens, prompt_tokens + num_prompt_tokens,
                         result.data()));

  free(prompt_tokens);
}