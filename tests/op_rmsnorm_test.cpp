#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "encoder.hpp"
#include "op.hpp"
#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class RmsNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t kSize = 4;
    x_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}));
    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}));

    for (size_t i = 0; i < kSize; i++) {
      (*x_)[i] = dis(gen);
      (*weight_)[i] = dis(gen);
    }

    transformer_ = std::make_unique<llama::Transformer<float>>(kChkPointPath);
    tokenizer_ = std::make_unique<llama::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  std::unique_ptr<llama::Tensor<float>> x_;
  std::unique_ptr<llama::Tensor<float>> weight_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama::Transformer<float>> transformer_;
  std::unique_ptr<llama::Tokenizer<float>> tokenizer_;
};

TEST_F(RmsNormTest, RmsNormTest) {
  const size_t kSize = 4;
  std::vector<float> expected_o(kSize);
  auto actual = llama::RmsNorm<float>::Compute(*x_, *weight_);
  reference::rmsnorm(expected_o.data(), x_->GetData(), weight_->GetData(),
                     kSize);

  EXPECT_TRUE(
      std::equal(expected_o.begin(), expected_o.end(), actual.GetData()));
}

TEST_F(RmsNormTest, ForwardTest) {
  const size_t kNumOfLayers = transformer_->GetConfig().NumLayers();
  const size_t kDim = transformer_->GetConfig().Dim();
  const auto& kWeights = transformer_->GetWeights();

  reference::Transformer ref_transformer;
  reference::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama::Encoder<float> encoder(*tokenizer_, kPrompt, true, false);
  auto content_row = kWeights.TokenEmbeddingTable() + kPos * kDim;
  std::copy(content_row, content_row + kDim,
            transformer_->GetRunState().X().GetData());

  std::copy(content_row, content_row + kDim, ref_run_state.x);

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    reference::rmsnorm(ref_run_state.xb, ref_run_state.x,
                       ref_weights.rms_att_weight + layer * kDim, kDim);

    llama::RmsNorm<float>::Compute(transformer_->GetRunState().X().GetData(),
                                   kWeights.RMSAttnWeight() + layer * kDim,
                                   kDim,
                                   transformer_->GetRunState().XB().GetData());
    EXPECT_TRUE(std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                           transformer_->GetRunState().XB().GetData()));
  }
}