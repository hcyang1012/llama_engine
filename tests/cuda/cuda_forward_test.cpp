#include <gtest/gtest.h>

#include <llama.hpp>
#include <op.hpp>
#include <references/reference_llama2.cpp>
#include <tests/test_utility.hpp>

#include "encoder.hpp"
class CUDAForwardTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";
};

TEST_F(CUDAForwardTest, Test) {
  const std::string kPrompt = "Who are you?";
  // Reference Forward Stage
  float *ref_logits = nullptr;
  {
    reference_llama2::Transformer ref_transformer;
    reference_llama2::build_transformer(&ref_transformer,
                                        kChkPointPath.c_str());

    reference_llama2::Tokenizer ref_tokenizer;
    reference_llama2::build_tokenizer(&ref_tokenizer, kTokenizerBinPath.c_str(),
                                      ref_transformer.config.vocab_size);

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(kPrompt.c_str()) + 3) *
                                       sizeof(int));  // +3 for '\0', ?BOS, ?EOS
    reference_llama2::encode(&ref_tokenizer, kPrompt.c_str(), 1, 0,
                             prompt_tokens, &num_prompt_tokens);
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
    ref_logits = reference_llama2::forward(&ref_transformer, token, pos);
  }

  // Forward Stage
  {
    const llama::LlamaConfig llama_config = {
        .checkpoint_path = kChkPointPath,
        .tokenizer_path = kTokenizerBinPath,
        .device_type = llama::DeviceType::CUDA};
    llama::Llama2<float> llama2(llama_config);
    auto &transformer = llama2.GetTransformer();
    const auto &tokenizer = llama2.GetTokenizer();
    auto encoder = llama::Encoder<float>(tokenizer, kPrompt, true, false,
                                         llama::SpecialTokensLlama2());
    auto result = encoder.PromptTokens();
    auto logits = transformer.Forward(result[0], 0);

    const float *logits_data =
        static_cast<float *>(logits.GetData()->GetBuffer());
    EXPECT_TRUE(llama::test::tensor_float_compare(
        logits_data, ref_logits, logits.GetShape().GetSize(), 1e-4));
  }
}