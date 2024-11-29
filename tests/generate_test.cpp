#include <gtest/gtest.h>

#include <op.hpp>

#include "encoder.hpp"
#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif
#include <llama.hpp>

#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class GenerateTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  float temperature_ = 0.0f;
  float topp_ = 0.9f;
  unsigned long long rng_seed_ = 1;
  const size_t kSteps = 256;

  llama::LlamaConfig llama2_config = {.checkpoint_path = kChkPointPath,
                                      .tokenizer_path = kTokenizerBinPath,
                                      .device_type = llama::DeviceType::CPU};
  std::unique_ptr<llama::LlamaModel<float>> llama2 =
      std::make_unique<llama::Llama2<float>>(llama2_config);
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
    const llama::RunConfig run_config = {
        .temperature = temperature_, .topp = topp_, .rng_seed = rng_seed_};

    std::string result = llama2->Generate(kPrompt, kSteps, run_config);
  }
}