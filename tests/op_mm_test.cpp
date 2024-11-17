#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "encoder.hpp"
#include "op.hpp"
#include "reference.cpp"
#include "tokenizer.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class MatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t d = 4;
    const size_t n = 5;
    left_ = std::make_unique<llama2::Tensor<float>>(llama2::Shape({n, d}));
    right_ = std::make_unique<llama2::Tensor<float>>(llama2::Shape({d, 1}));

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < d; j++) {
        (*left_)[i * d + j] = dis(gen);
      }
    }

    for (size_t i = 0; i < d; i++) {
      (*right_)[i] = dis(gen);
    }

    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
    tokenizer_ = std::make_unique<llama2::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  std::unique_ptr<llama2::Tensor<float>> left_;
  std::unique_ptr<llama2::Tensor<float>> right_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::Tokenizer<float>> tokenizer_;
};

TEST_F(MatMulTest, MatMulTest) {
  const size_t n = 5;
  const size_t d = 4;
  std::vector<float> expected_o(n);
  auto actual = llama2::MatMul<float>::Compute(*left_, *right_);
  reference::matmul(expected_o.data(), left_->GetData(), right_->GetData(), n,
                    d);

  EXPECT_TRUE(
      std::equal(expected_o.begin(), expected_o.end(), actual.GetData()));
}

TEST_F(MatMulTest, ForwardTest) {
  const auto& kConfig = transformer_->GetConfig();
  const size_t kNumOfLayers = kConfig.NumLayers();
  const size_t kDim = kConfig.Dim();
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

  const auto kKVDim = (kDim * kConfig.NumKVHeads()) / kConfig.NumHeads();

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    reference::rmsnorm(ref_run_state.xb, ref_run_state.x,
                       ref_weights.rms_att_weight + layer * kDim, kDim);

    llama2::RmsNorm<float>::Compute(transformer_->GetRunState().XB()->GetData(),
                                    transformer_->GetRunState().X()->GetData(),
                                    kWeights.RMSAttnWeight() + layer * kDim,
                                    kDim);
    EXPECT_TRUE(std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                           transformer_->GetRunState().XB()->GetData()));

    const auto kLayerOffset = layer * kConfig.SeqLen() * kKVDim;
    ref_run_state.k = ref_run_state.key_cache + kLayerOffset + kPos * kKVDim;
    ref_run_state.v = ref_run_state.value_cache + kLayerOffset + kPos * kKVDim;

    // Calculate Q

    llama2::MatMul<float>::Compute(transformer_->GetRunState().Q()->GetData(),
                                   transformer_->GetRunState().XB()->GetData(),
                                   kWeights.WQ() + layer * kDim * kDim, kDim,
                                   kDim);
    reference::matmul(ref_run_state.q, ref_run_state.xb,
                      ref_weights.wq + layer * kDim * kDim, kDim, kDim);

    EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                           transformer_->GetRunState().Q()->GetData()));

    // Calculate K
    const auto K = transformer_->GetRunState().KeyCache()->GetData() +
                   kLayerOffset + kPos * kKVDim;
    llama2::MatMul<float>::Compute(
        K, transformer_->GetRunState().XB()->GetData(),
        kWeights.WK() + layer * kDim * kKVDim, kDim, kKVDim);
    reference::matmul(ref_run_state.k, ref_run_state.xb,
                      ref_weights.wk + layer * kDim * kKVDim, kDim, kKVDim);
    EXPECT_TRUE(std::equal(ref_run_state.k, ref_run_state.k + kKVDim, K));

    // Calculate V
    const auto V = transformer_->GetRunState().ValueCache()->GetData() +
                   kLayerOffset + kPos * kKVDim;
    llama2::MatMul<float>::Compute(
        V, transformer_->GetRunState().XB()->GetData(),
        kWeights.WV() + layer * kDim * kKVDim, kDim, kKVDim);
    reference::matmul(ref_run_state.v, ref_run_state.xb,
                      ref_weights.wv + layer * kDim * kKVDim, kDim, kKVDim);
    EXPECT_TRUE(std::equal(ref_run_state.v, ref_run_state.v + kKVDim, V));
  }
}