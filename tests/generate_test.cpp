#include <gtest/gtest.h>

#include "encoder.hpp"
#include "op.hpp"
#include "reference.cpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class GenerateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    llama2::Transformer<float>::RunConfig run_config = {temperature_, topp_,
                                                        rng_seed_};
    transformer_ =
        std::make_unique<llama2::Transformer<float>>(kChkPointPath, run_config);
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

  // float temperature_ = 0.8f;
  float temperature_ = 0.0f;
  float topp_ = 0.9f;
  unsigned long long rng_seed_ = 1;
  const size_t kSteps = 256;
};

TEST_F(GenerateTest, Test) {
  const std::string kPrompt = "One day, Lily met a Shoggoth";
  // Reference Forward Stage
  float *ref_logits = nullptr;
  {
    reference::Transformer ref_transformer;
    reference::build_transformer(&ref_transformer, kChkPointPath.c_str());

    reference::Tokenizer ref_tokenizer;
    reference::build_tokenizer(&ref_tokenizer, kTokenizerBinPath.c_str(),
                               ref_transformer.config.vocab_size);

    reference::Sampler sampler;
    reference::build_sampler(&sampler, ref_transformer.config.vocab_size,
                             temperature_, topp_, rng_seed_);

    std::cout << "Reference Generation:" << std::endl;
    reference::generate(&ref_transformer, &ref_tokenizer, &sampler,
                        kPrompt.c_str(), kSteps);
  }
  {
    std::cout << "LLAMA2 Generation:" << std::endl;
    transformer_->Generate(*tokenizer_, kPrompt, kSteps);
  }
}