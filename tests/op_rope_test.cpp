#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "encoder.hpp"
#include "op.hpp"
#include "references/reference_llama2.cpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class RopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
    tokenizer_ = std::make_unique<llama2::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void reference_rope(const float *input, const int position, float *output) {
    const auto dim = transformer_->GetConfig().Dim();
    const auto kv_dim = (transformer_->GetConfig().Dim() *
                         transformer_->GetConfig().NumKVHeads()) /
                        transformer_->GetConfig().NumHeads();
    const auto head_size = dim / transformer_->GetConfig().NumHeads();
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float theta = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = position * theta;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int v = 0; v < rotn; v++) {
        float v0 = input[i];
        float v1 = input[i + 1];
        output[i] = v0 * fcr - v1 * fci;
        output[i + 1] = v0 * fci + v1 * fcr;
      }
    }
  }

  void TearDown() override {}

  std::unique_ptr<llama2::Tensor<float>> x_;
  std::unique_ptr<llama2::Tensor<float>> weight_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::Tokenizer<float>> tokenizer_;
};

TEST_F(RopeTest, ForwardTest) {
  const size_t kNumOfLayers = transformer_->GetConfig().NumLayers();
  const size_t kDim = transformer_->GetConfig().Dim();
  const auto &kWeights = transformer_->GetWeights();

  reference::Transformer ref_transformer;
  reference::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama2::Encoder<float> encoder(*tokenizer_, kPrompt, true, false);
  auto content_row = kWeights.TokenEmbeddingTable() + kPos * kDim;
  std::copy(content_row, content_row + kDim,
            transformer_->GetRunState().X().GetData());

  std::copy(content_row, content_row + kDim, ref_run_state.x);

  const auto &kRefConfig = ref_transformer.config;
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

          transformer_->GetRunState().X().GetData(),
          kWeights.RMSAttnWeight() + layer * kDim, kDim,
          transformer_->GetRunState().XB().GetData());

      EXPECT_TRUE(std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                             transformer_->GetRunState().XB().GetData()));
    }

    const int kRefLayerOffset = layer * kRefConfig.seq_len * kRefKVDim;
    ref_run_state.k =
        ref_run_state.key_cache + kRefLayerOffset + kPos * kRefKVDim;
    ref_run_state.v =
        ref_run_state.value_cache + kRefLayerOffset + kPos * kRefKVDim;

    // Calculate QKV
    {
      // qkv matmuls for this position
      reference::matmul(ref_run_state.q, ref_run_state.xb,
                        ref_weights.wq + layer * kDim * kDim, kDim, kDim);
      reference::matmul(ref_run_state.k, ref_run_state.xb,
                        ref_weights.wk + layer * kDim * kRefKVDim, kDim,
                        kRefKVDim);
      reference::matmul(ref_run_state.v, ref_run_state.xb,
                        ref_weights.wv + layer * kDim * kRefKVDim, kDim,
                        kRefKVDim);

      llama2::MatMul<float>::Compute(
          kWeights.WQ(layer).ReShape(llama2::Shape({kDim, kDim})),
          transformer_->GetRunState().XB(), transformer_->GetRunState().Q());

      auto K = transformer_->GetRunState().K(layer, kPos).ReShape({kKVDim});
      llama2::MatMul<float>::Compute(
          kWeights.WK(layer).ReShape(llama2::Shape({kDim, kRefKVDim})),
          transformer_->GetRunState().XB(), K);

      auto V = transformer_->GetRunState().V(layer, kPos).ReShape({kKVDim});
      llama2::MatMul<float>::Compute(
          kWeights.WV(layer).ReShape(llama2::Shape({kDim, kRefKVDim})),
          transformer_->GetRunState().XB(), V);

      EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                             transformer_->GetRunState().Q().GetData()));
      EXPECT_TRUE(
          std::equal(ref_run_state.k, ref_run_state.k + kRefKVDim,
                     transformer_->GetRunState().K(layer, kPos).GetData()));
      EXPECT_TRUE(
          std::equal(ref_run_state.v, ref_run_state.v + kRefKVDim,
                     transformer_->GetRunState().V(layer, kPos).GetData()));
    }

    const size_t kRefHeadSize = kDim / kRefConfig.n_heads;
    // RoPE
    {
      for (int i = 0; i < kDim; i += 2) {
        int head_dim = i % kRefHeadSize;
        float theta = 1.0f / powf(10000.0f, head_dim / (float)kRefHeadSize);
        float val = kPos * theta;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn =
            i < kRefKVDim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
          float *vec = v == 0 ? ref_run_state.q
                              : ref_run_state.k;  // the vector to rotate
          float v0 = vec[i];
          float v1 = vec[i + 1];
          vec[i] = v0 * fcr - v1 * fci;
          vec[i + 1] = v0 * fci + v1 * fcr;
        }
      }
      auto Q = transformer_->GetRunState().Q();
      auto K = transformer_->GetRunState().K(0, kPos);
      llama2::RoPE<float>::Compute(kPos, transformer_->GetConfig(), Q, K);

      EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                             transformer_->GetRunState().Q().GetData()));
    }
  }
}