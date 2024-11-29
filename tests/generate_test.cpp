#include <gtest/gtest.h>

#include <op.hpp>

#include "encoder.hpp"
#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class GenerateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    llama::Transformer<float>::RunConfig run_config = {temperature_, topp_,
                                                       rng_seed_};
    transformer_ = std::make_unique<llama::Transformer<float>>(
        kChkPointPath, run_config, *op_set_, llama::SpecialTokensLlama2());
    tokenizer_ = std::make_unique<llama::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama::Transformer<float>> transformer_;
  std::unique_ptr<llama::Tokenizer<float>> tokenizer_;
  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(llama::DeviceType::CPU);

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
    reference_llama2::Transformer ref_transformer;
    reference_llama2::build_transformer(&ref_transformer,
                                        kChkPointPath.c_str());

    reference_llama2::Tokenizer ref_tokenizer;
    reference_llama2::build_tokenizer(&ref_tokenizer, kTokenizerBinPath.c_str(),
                                      ref_transformer.config.vocab_size);

    reference_llama2::Sampler sampler;
    reference_llama2::build_sampler(&sampler, ref_transformer.config.vocab_size,
                                    temperature_, topp_, rng_seed_);

    std::cout << "Reference Generation:" << std::endl;
    reference_llama2::generate(&ref_transformer, &ref_tokenizer, &sampler,
                               kPrompt.c_str(), kSteps);
  }
  {
    std::cout << "LLAMA2 Generation:" << std::endl;
    transformer_->Generate(*tokenizer_, kPrompt, kSteps);
  }
}