#include <gtest/gtest.h>

// For random number generation
#include <op.hpp>
#include <random>

#include "encoder.hpp"
#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif
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

    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({d, n}),
                                                     op_set_->GetDeviceType());
    input_ = std::make_unique<llama::Tensor<float>>(llama::Shape({d}),
                                                    op_set_->GetDeviceType());

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < d; j++) {
        (*weight_)[i * d + j] = dis(gen);
      }
    }

    for (size_t i = 0; i < d; i++) {
      (*input_)[i] = dis(gen);
    }

    transformer_ =
        std::make_unique<llama::Transformer<float>>(kChkPointPath, *op_set_);
    tokenizer_ = std::make_unique<llama::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  // (N, D) * (D, 1) = (N, 1)
  // Where
  // Shape of weight:
  //  [0]: D
  //  [1]: N
  // Shape of X:
  //  [0]: 1
  //  [1]: D
  const size_t d = 4;
  const size_t n = 5;

  std::unique_ptr<llama::Tensor<float>> weight_;
  std::unique_ptr<llama::Tensor<float>> input_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama::Transformer<float>> transformer_;
  std::unique_ptr<llama::Tokenizer<float>> tokenizer_;

  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(llama::DeviceType::CPU);
};

TEST_F(MatMulTest, MatMulTest) {
  std::vector<float> expected_o(n);
  llama::Tensor<float> actual({n}, op_set_->GetDeviceType());
  op_set_->MatMul<float>(*weight_, *input_, actual);
  reference_llama2::matmul(expected_o.data(), input_->GetData(),
                           weight_->GetData(), d, n);

  EXPECT_TRUE(
      std::equal(expected_o.begin(), expected_o.end(), actual.GetData()));
}

TEST_F(MatMulTest, ForwardTest) {
  const auto& kConfig = transformer_->GetConfig();
  const size_t kNumOfLayers = kConfig.NumLayers();
  const size_t kDim = kConfig.Dim();
  const auto& kWeights = transformer_->GetWeights();

  reference_llama2::Transformer ref_transformer;
  reference_llama2::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama::Encoder<float> encoder(*tokenizer_, kPrompt, true, false);
  auto content_row = kWeights.TokenEmbeddingTable() + kPos * kDim;
  std::copy(content_row, content_row + kDim,
            transformer_->GetRunState().X().GetData());

  std::copy(content_row, content_row + kDim, ref_run_state.x);

  const auto kKVDim = (kDim * kConfig.NumKVHeads()) / kConfig.NumHeads();

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                              ref_weights.rms_att_weight + layer * kDim, kDim);

    op_set_->RmsNorm<float>(transformer_->GetRunState().X(),
                            kWeights.RMSAttnWeight(layer),
                            transformer_->GetRunState().XB());

    EXPECT_TRUE(std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                           transformer_->GetRunState().XB().GetData()));

    const auto kLayerOffset = layer * kConfig.SeqLen() * kKVDim;
    ref_run_state.k = ref_run_state.key_cache + kLayerOffset + kPos * kKVDim;
    ref_run_state.v = ref_run_state.value_cache + kLayerOffset + kPos * kKVDim;

    // Calculate Q
    op_set_->MatMul<float>(
        kWeights.WQ(layer).ReShape(llama::Shape({kDim, kDim})),
        transformer_->GetRunState().XB(), transformer_->GetRunState().Q());

    reference_llama2::matmul(ref_run_state.q, ref_run_state.xb,
                             ref_weights.wq + layer * kDim * kDim, kDim, kDim);

    EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                           transformer_->GetRunState().Q().GetData()));

    // Calculate K
    auto K = transformer_->GetRunState().K(layer, kPos).ReShape({kKVDim});
    op_set_->MatMul<float>(
        kWeights.WK(layer).ReShape(llama::Shape({kDim, kKVDim})),
        transformer_->GetRunState().XB(), K);
    reference_llama2::matmul(ref_run_state.k, ref_run_state.xb,
                             ref_weights.wk + layer * kDim * kKVDim, kDim,
                             kKVDim);
    EXPECT_TRUE(
        std::equal(ref_run_state.k, ref_run_state.k + kKVDim, K.GetData()));

    // Calculate V
    auto V = transformer_->GetRunState().V(layer, kPos).ReShape({kKVDim});
    op_set_->MatMul<float>(
        kWeights.WV(layer).ReShape(llama::Shape({kDim, kKVDim})),
        transformer_->GetRunState().XB(), V);
    reference_llama2::matmul(ref_run_state.v, ref_run_state.xb,
                             ref_weights.wv + layer * kDim * kKVDim, kDim,
                             kKVDim);
    EXPECT_TRUE(
        std::equal(ref_run_state.v, ref_run_state.v + kKVDim, V.GetData()));
  }
}