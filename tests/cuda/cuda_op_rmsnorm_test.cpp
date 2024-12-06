#include <gtest/gtest.h>

#include <llama.hpp>
#include <op.hpp>
#include <random>
#include <references/reference_llama2.cpp>
class CUDARmsNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t kSize = 4;
    x_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                llama::DeviceType::CPU);
    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                     llama::DeviceType::CPU);

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
      .device_type = llama::DeviceType::CUDA};
  std::unique_ptr<llama::Llama2<float>> llama2_ =
      std::make_unique<llama::Llama2<float>>(kLlamaConfig);

  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(kLlamaConfig.device_type);
};

TEST_F(CUDARmsNormTest, RmsNormTest) {
  const size_t kSize = 4;
  std::vector<float> expected_o(kSize);
  llama::Tensor<float> actual(x_->GetShape(), kLlamaConfig.device_type);
  llama::Tensor<float> device_x(x_->GetShape(), kLlamaConfig.device_type);
  llama::Tensor<float> device_weight(weight_->GetShape(),
                                     kLlamaConfig.device_type);

  llama::DeviceFactory::GetDevice(kLlamaConfig.device_type)
      .GetMemcpy()
      .Copy(*device_x.GetData(), *x_->GetData());
  llama::DeviceFactory::GetDevice(kLlamaConfig.device_type)
      .GetMemcpy()
      .Copy(*device_weight.GetData(), *weight_->GetData());

  auto op_set_ = llama::CreateOpSet(llama::DeviceType::CUDA);
  op_set_->RmsNorm<float>(device_x, device_weight, actual);

  reference_llama2::rmsnorm(
      expected_o.data(), static_cast<const float*>(x_->GetData()->GetBuffer()),
      static_cast<const float*>(weight_->GetData()->GetBuffer()), kSize);

  auto out_buffer_host = actual.Dump();
  auto out_buffer_host_ptr =
      static_cast<float*>(out_buffer_host.GetData()->GetBuffer());
  bool equal = true;
  for (size_t i = 0; i < kSize; i++) {
    if (std::abs(expected_o[i] - out_buffer_host_ptr[i]) > 1e-5) {
      equal = false;
      break;
    }
  }
  EXPECT_TRUE(equal) << "Mismatch";
}

TEST_F(CUDARmsNormTest, ForwardTest) {
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
  auto embed =
      static_cast<float*>(kWeights.TokenEmbeddingTable()->GetBuffer()) +
      kPos * kDim;
  llama::DeviceFactory::GetDevice(kLlamaConfig.device_type)
      .GetMemcpy()
      .Copy(*(llama2_->GetTransformer().GetRunState().X().GetData()),
            (void*)embed, kDim * sizeof(float));
  float* content_row =
      ref_transformer.weights.token_embedding_table + kPos * kDim;
  memcpy(ref_run_state.x, content_row, kDim * sizeof(*ref_run_state.x));

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                              ref_weights.rms_att_weight + layer * kDim, kDim);

    op_set_->RmsNorm<float>(transformer.GetRunState().X(),
                            kWeights.RMSAttnWeight(layer),
                            transformer.GetRunState().XB());

    auto out_buffer_host = llama::Tensor<float>(
        transformer.GetRunState().XB().GetShape(), llama::DeviceType::CPU);
    llama::DeviceFactory::GetDevice(llama::DeviceType::CUDA)
        .GetMemcpy()
        .Copy(*(out_buffer_host.GetData()),
              *(transformer.GetRunState().XB().GetData()));

    bool equal = true;
    for (size_t i = 0; i < kDim; i++) {
      if (std::abs(ref_run_state.xb[i] - out_buffer_host[i]) > 1e-5) {
        equal = false;
        break;
      }
    }
    EXPECT_TRUE(equal) << "Mismatch at layer " << layer;
  }
}