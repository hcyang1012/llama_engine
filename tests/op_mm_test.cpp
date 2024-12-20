#include <gtest/gtest.h>

#include <llama.hpp>
#include <op.hpp>
#include <random>

#include "references/reference_llama2.cpp"
class MatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({d, n}),
                                                     kLlamaConfig.device_type);
    input_ = std::make_unique<llama::Tensor<float>>(llama::Shape({d}),
                                                    kLlamaConfig.device_type);

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < d; j++) {
        (*weight_)[i * d + j] = dis(gen);
      }
    }

    for (size_t i = 0; i < d; i++) {
      (*input_)[i] = dis(gen);
    }
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

  const llama::LlamaConfig kLlamaConfig = {
      .checkpoint_path = kChkPointPath.c_str(),
      .tokenizer_path = kTokenizerBinPath.c_str(),
      .device_type = llama::DeviceType::CPU};
  std::unique_ptr<llama::Llama2<float>> llama2_ =
      std::make_unique<llama::Llama2<float>>(kLlamaConfig);
  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(kLlamaConfig.device_type);
};

TEST_F(MatMulTest, MatMulTest) {
  std::vector<float> expected_o(n);
  llama::Tensor<float> actual({n}, kLlamaConfig.device_type);
  op_set_->MatMul<float>(*weight_, *input_, actual);
  reference_llama2::matmul(
      expected_o.data(), static_cast<float*>(input_->GetData()->GetBuffer()),
      static_cast<float*>(weight_->GetData()->GetBuffer()), d, n);

  EXPECT_TRUE(std::equal(expected_o.begin(), expected_o.end(),
                         static_cast<float*>(actual.GetData()->GetBuffer())));
}

TEST_F(MatMulTest, ForwardTest) {
  auto& transformer = llama2_->GetTransformer();
  const auto& tokenizer = llama2_->GetTokenizer();
  const auto& kConfig = transformer.GetConfig();
  const size_t kNumOfLayers = kConfig.NumLayers();
  const size_t kDim = kConfig.Dim();
  const auto& kWeights = transformer.GetWeights();

  reference_llama2::Transformer ref_transformer;
  reference_llama2::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama::Encoder<float> encoder(tokenizer, kPrompt, true, false,
                                llama::SpecialTokensLlama2());
  auto content_row =
      static_cast<float*>(kWeights.TokenEmbeddingTable()->GetBuffer()) +
      kPos * kDim;

  llama::GetMemcpy(kLlamaConfig.device_type)
      .Copy(*transformer.GetRunState().X().GetData(), (void*)content_row,
            kDim * sizeof(float));

  std::copy(content_row, content_row + kDim, ref_run_state.x);

  const auto kKVDim = (kDim * kConfig.NumKVHeads()) / kConfig.NumHeads();

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                              ref_weights.rms_att_weight + layer * kDim, kDim);

    op_set_->RmsNorm<float>(transformer.GetRunState().X(),
                            kWeights.RMSAttnWeight(layer),
                            transformer.GetRunState().XB());

    EXPECT_TRUE(
        std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                   static_cast<float*>(
                       transformer.GetRunState().XB().GetData()->GetBuffer())));

    const auto kLayerOffset = layer * kConfig.SeqLen() * kKVDim;
    ref_run_state.k = ref_run_state.key_cache + kLayerOffset + kPos * kKVDim;
    ref_run_state.v = ref_run_state.value_cache + kLayerOffset + kPos * kKVDim;

    // Calculate Q
    op_set_->MatMul<float>(
        kWeights.WQ(layer).ReShape(llama::Shape({kDim, kDim})),
        transformer.GetRunState().XB(), transformer.GetRunState().Q());

    reference_llama2::matmul(ref_run_state.q, ref_run_state.xb,
                             ref_weights.wq + layer * kDim * kDim, kDim, kDim);

    EXPECT_TRUE(
        std::equal(ref_run_state.q, ref_run_state.q + kDim,
                   static_cast<float*>(
                       transformer.GetRunState().Q().GetData()->GetBuffer())));

    // Calculate K
    auto K = transformer.GetRunState().K(layer, kPos).ReShape({kKVDim});
    op_set_->MatMul<float>(
        kWeights.WK(layer).ReShape(llama::Shape({kDim, kKVDim})),
        transformer.GetRunState().XB(), K);
    reference_llama2::matmul(ref_run_state.k, ref_run_state.xb,
                             ref_weights.wk + layer * kDim * kKVDim, kDim,
                             kKVDim);
    EXPECT_TRUE(std::equal(ref_run_state.k, ref_run_state.k + kKVDim,
                           static_cast<float*>(K.GetData()->GetBuffer())));

    // Calculate V
    auto V = transformer.GetRunState().V(layer, kPos).ReShape({kKVDim});
    op_set_->MatMul<float>(
        kWeights.WV(layer).ReShape(llama::Shape({kDim, kKVDim})),
        transformer.GetRunState().XB(), V);
    reference_llama2::matmul(ref_run_state.v, ref_run_state.xb,
                             ref_weights.wv + layer * kDim * kKVDim, kDim,
                             kKVDim);
    EXPECT_TRUE(std::equal(ref_run_state.v, ref_run_state.v + kKVDim,
                           static_cast<float*>(V.GetData()->GetBuffer())));
  }
}