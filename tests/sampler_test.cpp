#include "sampler.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "reference.cpp"
#include "transformer.hpp"
class SamplerTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    build_transformer(&ref_transformer_, kChkPointPath.c_str());
    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);

    build_sampler(&ref_sampler_, ref_transformer_.config.vocab_size,
                  kTemperature, kTopP, kRngSeed);
    sampler_ = std::make_unique<llama2::Sampler>(
        transformer_->GetConfig().VocabSize(), kTemperature, kTopP, kRngSeed);
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  const std::string kChkPointPath = "stories15M.bin";
  const float kTemperature = 0.0f;
  const float kTopP = 0.9f;
  const uint64_t kRngSeed = 0;

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::Sampler> sampler_;

  reference::Transformer ref_transformer_;
  reference::Sampler ref_sampler_;
};

TEST_F(SamplerTest, VocabSizeTest) {
  EXPECT_EQ(sampler_->VocabSize(), ref_sampler_.vocab_size);
}

TEST_F(SamplerTest, TemperatureTest) {
  EXPECT_EQ(sampler_->Temperature(), ref_sampler_.temperature);
}

TEST_F(SamplerTest, TopPTest) {
  EXPECT_EQ(sampler_->TopP(), ref_sampler_.topp);
}

TEST_F(SamplerTest, RngSeedTest) {
  EXPECT_EQ(sampler_->RngState(), ref_sampler_.rng_state);
}

TEST_F(SamplerTest, ProbIndicesTest) {
  const auto& kProbIndices = sampler_->ProbIndices();
  EXPECT_EQ(kProbIndices.size(), ref_sampler_.vocab_size);
}