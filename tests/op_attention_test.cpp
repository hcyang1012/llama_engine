#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "encoder.hpp"
#include "op.hpp"
#include "reference.cpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t kSize = 4;
    x_ = std::make_unique<llama2::Tensor<float>>(llama2::Shape({kSize}));
    weight_ = std::make_unique<llama2::Tensor<float>>(llama2::Shape({kSize}));

    for (size_t i = 0; i < kSize; i++) {
      (*x_)[i] = dis(gen);
      (*weight_)[i] = dis(gen);
    }

    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
    tokenizer_ = std::make_unique<llama2::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  std::unique_ptr<llama2::Tensor<float>> x_;
  std::unique_ptr<llama2::Tensor<float>> weight_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::Tokenizer<float>> tokenizer_;
};

TEST_F(AttentionTest, ForwardTest) {
  const size_t kNumOfLayers = transformer_->GetConfig().NumLayers();
  const size_t kDim = transformer_->GetConfig().Dim();
  const auto& kWeights = transformer_->GetWeights();

  reference::Transformer ref_transformer;
  reference::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama2::Encoder<float> encoder(*tokenizer_, kPrompt, true, false);
  auto content_row = kWeights.TokenEmbeddingTable() + kPos * kDim;
  std::copy(content_row, content_row + kDim,
            transformer_->GetRunState().X()->GetData());

  std::copy(content_row, content_row + kDim, ref_run_state.x);

  const auto& kRefConfig = ref_transformer.config;
  const size_t kRefKVDim =
      (kRefConfig.dim * kRefConfig.n_kv_heads) / kRefConfig.n_heads;

  const size_t kKVDim = (kDim * transformer_->GetConfig().NumKVHeads()) /
                        transformer_->GetConfig().NumHeads();

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    // RMSNorm
    {
      reference::rmsnorm(ref_run_state.xb, ref_run_state.x,
                         ref_weights.rms_att_weight + layer * kDim, kDim);

      llama2::RmsNorm<float>::Compute(
          transformer_->GetRunState().XB()->GetData(),
          transformer_->GetRunState().X()->GetData(),
          kWeights.RMSAttnWeight() + layer * kDim, kDim);

      EXPECT_TRUE(std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                             transformer_->GetRunState().XB()->GetData()));
    }

    // Calculate QKV
    {
      const int kRefLayerOffset = layer * kRefConfig.seq_len * kRefKVDim;
      ref_run_state.k =
          ref_run_state.key_cache + kRefLayerOffset + kPos * kRefKVDim;
      ref_run_state.v =
          ref_run_state.value_cache + kRefLayerOffset + kPos * kRefKVDim;

      // qkv matmuls for this position
      reference::matmul(ref_run_state.q, ref_run_state.xb,
                        ref_weights.wq + layer * kDim * kDim, kDim, kDim);
      reference::matmul(ref_run_state.k, ref_run_state.xb,
                        ref_weights.wk + layer * kDim * kRefKVDim, kDim,
                        kRefKVDim);
      reference::matmul(ref_run_state.v, ref_run_state.xb,
                        ref_weights.wv + layer * kDim * kRefKVDim, kDim,
                        kRefKVDim);

      transformer_->GetRunState().UpdateKV(layer, kPos);
      llama2::MatMul<float>::Compute(
          transformer_->GetRunState().Q()->GetData(),
          transformer_->GetRunState().XB()->GetData(),
          kWeights.WQ() + layer * kDim * kDim, kDim, kDim);

      llama2::MatMul<float>::Compute(
          transformer_->GetRunState().K(),
          transformer_->GetRunState().XB()->GetData(),
          kWeights.WK() + layer * kDim * kKVDim, kDim, kKVDim);

      llama2::MatMul<float>::Compute(
          transformer_->GetRunState().V(),
          transformer_->GetRunState().XB()->GetData(),
          kWeights.WV() + layer * kDim * kKVDim, kDim, kKVDim);

      EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                             transformer_->GetRunState().Q()->GetData()));
      EXPECT_TRUE(std::equal(ref_run_state.k, ref_run_state.k + kRefKVDim,
                             transformer_->GetRunState().K()));
      EXPECT_TRUE(std::equal(ref_run_state.v, ref_run_state.v + kRefKVDim,
                             transformer_->GetRunState().V()));
    }

    // Attention
    {
      for (size_t head_idx = 0; head_idx < kRefConfig.n_heads; head_idx++) {
        // float *q = ref_run_state.q + head_idx *
      }
    }
  }
}