#include <gtest/gtest.h>

// For random number generation
#include <llama.hpp>
#include <op.hpp>
#include <random>
#include <references/reference_llama2.cpp>
class RmsNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t kSize = 4;
    x_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                kLlamaConfig.device_type);
    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                     kLlamaConfig.device_type);

    for (size_t i = 0; i < kSize; i++) {
      (*x_)[i] = dis(gen);
      (*weight_)[i] = dis(gen);
    }
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  std::unique_ptr<llama::Tensor<float>> x_;
  std::unique_ptr<llama::Tensor<float>> weight_;

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

TEST_F(RmsNormTest, RmsNormTest) {
  const size_t kSize = 4;
  std::vector<float> expected_o(kSize);
  llama::Tensor<float> actual(x_->GetShape(), kLlamaConfig.device_type);

  auto op_set_ = llama::CreateOpSet(llama::DeviceType::CPU);
  op_set_->RmsNorm<float>(*x_, *weight_, actual);
  reference_llama2::rmsnorm(expected_o.data(), x_->GetData(),
                            weight_->GetData(), kSize);

  EXPECT_TRUE(
      std::equal(expected_o.begin(), expected_o.end(), actual.GetData()));
}

TEST_F(RmsNormTest, ForwardTest) {
  auto& transformer = llama2_->GetTransformer();
  const auto& tokenizer = llama2_->GetTokenizer();
  const size_t kNumOfLayers = transformer.GetConfig().NumLayers();
  const size_t kDim = transformer.GetConfig().Dim();
  const auto& kWeights = transformer.GetWeights();

  reference_llama2::Transformer ref_transformer;
  reference_llama2::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama::Encoder<float> encoder(tokenizer, kPrompt, true, false,
                                llama::SpecialTokensLlama2());
  auto content_row = kWeights.TokenEmbeddingTable() + kPos * kDim;
  std::copy(content_row, content_row + kDim,
            transformer.GetRunState().X().GetData());

  std::copy(content_row, content_row + kDim, ref_run_state.x);

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                              ref_weights.rms_att_weight + layer * kDim, kDim);

    op_set_->RmsNorm<float>(transformer.GetRunState().X(),
                            kWeights.RMSAttnWeight(layer),
                            transformer.GetRunState().XB());

    EXPECT_TRUE(std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                           transformer.GetRunState().XB().GetData()));
  }
}