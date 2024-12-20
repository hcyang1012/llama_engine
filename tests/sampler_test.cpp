#include "sampler.hpp"

#include <gtest/gtest.h>

#include <llama.hpp>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "references/reference_llama2.cpp"
class SamplerTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    build_transformer(&ref_transformer_, kChkPointPath.c_str());
    build_sampler(&ref_sampler_, ref_transformer_.config.vocab_size,
                  kTemperature, kTopP, kRngSeed);

    auto& transformer = llama2_->GetTransformer();
    sampler_ = std::make_unique<llama::Sampler>(
        transformer.GetConfig().VocabSize(), kTemperature, kTopP, kRngSeed);
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";
  const float kTemperature = 0.0f;
  const float kTopP = 0.9f;
  const uint64_t kRngSeed = 0;

  std::unique_ptr<llama::Sampler> sampler_;
  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(llama::DeviceType::CPU);
  const llama::LlamaConfig kLlamaConfig = {
      .checkpoint_path = kChkPointPath.c_str(),
      .tokenizer_path = kTokenizerBinPath.c_str(),
      .device_type = llama::DeviceType::CPU};
  std::unique_ptr<llama::Llama2<float>> llama2_ =
      std::make_unique<llama::Llama2<float>>(kLlamaConfig);

  reference_llama2::Transformer ref_transformer_;
  reference_llama2::Sampler ref_sampler_;
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

TEST_F(SamplerTest, ProbIndicesTest) {
  const auto& kProbIndices = sampler_->ProbIndices();
  EXPECT_EQ(kProbIndices.size(), ref_sampler_.vocab_size);
}

TEST_F(SamplerTest, ProbIdxSortTest) {
  std::vector<float> input;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  std::vector<llama::Sampler::ProbIdx> prob_indices;
  for (int i = 0; i < 10; i++) {
    input.push_back(dis(gen));
    llama::Sampler::ProbIdx prob_idx;
    prob_idx.prob = input[i];
    prob_idx.idx = i;
    prob_indices.push_back(prob_idx);
  }

  std::sort(prob_indices.begin(), prob_indices.end());

  std::vector<reference_llama2::ProbIndex> ref_prob_indices;
  for (int i = 0; i < 10; i++) {
    reference_llama2::ProbIndex ref_prob_idx;
    ref_prob_idx.prob = input[i];
    ref_prob_idx.index = i;
    ref_prob_indices.push_back(ref_prob_idx);
  }
  qsort(ref_prob_indices.data(), ref_prob_indices.size(),
        sizeof(reference_llama2::ProbIndex), reference_llama2::compare);

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(prob_indices[i].prob, ref_prob_indices[i].prob);
    EXPECT_EQ(prob_indices[i].idx, ref_prob_indices[i].index);
  }
}

TEST_F(SamplerTest, SampleTest) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  const size_t kNumOfLogits = sampler_->VocabSize();
  auto logits = llama::GetMemoryAllocator(llama::DeviceType::CPU)
                    .Allocate(kNumOfLogits * sizeof(float));
  auto logits_buf = static_cast<float*>(logits->GetBuffer());
  for (size_t i = 0; i < kNumOfLogits; i++) {
    logits_buf[i] = dis(gen);
  }

  llama::Tensor<float> logits_tensor(logits, {kNumOfLogits},
                                     op_set_->GetDeviceType());
  const int kNext = sampler_->Sample(logits_tensor);

  int ref_next = sample(&ref_sampler_, logits_buf);
}