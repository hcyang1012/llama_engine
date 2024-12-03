#include <gtest/gtest.h>

#include <llama.hpp>
#include <op.hpp>
#include <random>
#include <references/reference_llama2.cpp>
#include <test_utility.hpp>
class CUDAMatMulTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({d, n}),
                                                     llama::DeviceType::CPU);
    input_ = std::make_unique<llama::Tensor<float>>(llama::Shape({d}),
                                                    llama::DeviceType::CPU);

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
      .device_type = llama::DeviceType::CUDA};
  std::unique_ptr<llama::Llama2<float>> llama2_ =
      std::make_unique<llama::Llama2<float>>(kLlamaConfig);
  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(kLlamaConfig.device_type);
};

TEST_F(CUDAMatMulTest, CUDAMatMulTest) {
  std::vector<float> expected_o(n);

  llama::Tensor<float> device_weight(weight_->GetShape(),
                                     kLlamaConfig.device_type);
  llama::Tensor<float> device_input(input_->GetShape(),
                                    kLlamaConfig.device_type);
  llama::Tensor<float> actual({n}, kLlamaConfig.device_type);

  llama::DeviceFactory::GetDevice(kLlamaConfig.device_type)
      .GetMemcpy()
      .Copy(*device_weight.GetData(), *weight_->GetData());
  llama::DeviceFactory::GetDevice(kLlamaConfig.device_type)
      .GetMemcpy()
      .Copy(*device_input.GetData(), *input_->GetData());

  op_set_->MatMul<float>(device_weight, device_input, actual);
  reference_llama2::matmul(
      expected_o.data(), static_cast<float*>(input_->GetData()->GetBuffer()),
      static_cast<float*>(weight_->GetData()->GetBuffer()), d, n);

  llama::Tensor<float> out_buffer_host(actual.GetShape(),
                                       llama::DeviceType::CPU);
  llama::DeviceFactory::GetDevice(llama::DeviceType::CUDA)
      .GetMemcpy()
      .Copy(*out_buffer_host.GetData(), *actual.GetData());

  bool equal = true;
  for (size_t i = 0; i < expected_o.size(); i++) {
    if (std::abs(expected_o[i] - out_buffer_host[i]) > 1e-5) {
      equal = false;
      break;
    }
  }
  EXPECT_TRUE(equal) << "Mismatch in MatMul";
}

TEST_F(CUDAMatMulTest, ForwardTest) {
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

  const auto kKVDim = (kDim * kConfig.NumKVHeads()) / kConfig.NumHeads();

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    const auto kLayerOffset = layer * kConfig.SeqLen() * kKVDim;
    {
      reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                                ref_weights.rms_att_weight + layer * kDim,
                                kDim);

      ref_run_state.k = ref_run_state.key_cache + kLayerOffset + kPos * kKVDim;
      ref_run_state.v =
          ref_run_state.value_cache + kLayerOffset + kPos * kKVDim;

      reference_llama2::matmul(ref_run_state.q, ref_run_state.xb,
                               ref_weights.wq + layer * kDim * kDim, kDim,
                               kDim);
      reference_llama2::matmul(ref_run_state.k, ref_run_state.xb,
                               ref_weights.wk + layer * kDim * kKVDim, kDim,
                               kKVDim);
      reference_llama2::matmul(ref_run_state.v, ref_run_state.xb,
                               ref_weights.wv + layer * kDim * kKVDim, kDim,
                               kKVDim);
    }
    {
      op_set_->RmsNorm<float>(transformer.GetRunState().X(),
                              kWeights.RMSAttnWeight(layer),
                              transformer.GetRunState().XB());

      // Calculate Q
      op_set_->MatMul<float>(
          kWeights.WQ(layer).ReShape(llama::Shape({kDim, kDim})),
          transformer.GetRunState().XB(), transformer.GetRunState().Q());

      llama::Tensor<float> Q_host(transformer.GetRunState().Q().GetShape(),
                                  llama::DeviceType::CPU);
      llama::DeviceFactory::GetDevice(llama::DeviceType::CUDA)
          .GetMemcpy()
          .Copy(*Q_host.GetData(), *transformer.GetRunState().Q().GetData());

      EXPECT_TRUE(llama::test::tensor_float_compare(
          ref_run_state.q, static_cast<float*>(Q_host.GetData()->GetBuffer()),
          kDim, 1e-5))
          << "Mismatch in Q";

      // Calculate K
      auto K = transformer.GetRunState().K(layer, kPos).ReShape({kKVDim});
      op_set_->MatMul<float>(
          kWeights.WK(layer).ReShape(llama::Shape({kDim, kKVDim})),
          transformer.GetRunState().XB(), K);

      llama::Tensor<float> K_host(K.GetShape(), llama::DeviceType::CPU);
      llama::DeviceFactory::GetDevice(llama::DeviceType::CUDA)
          .GetMemcpy()
          .Copy(*K_host.GetData(), *K.GetData(), K.GetDataBytesSize());

      EXPECT_TRUE(llama::test::tensor_float_compare(
          ref_run_state.k, static_cast<float*>(K_host.GetData()->GetBuffer()),
          kKVDim, 1e-5))
          << "Mismatch in K";

      // Calculate V
      auto V = transformer.GetRunState().V(layer, kPos).ReShape({kKVDim});
      op_set_->MatMul<float>(
          kWeights.WV(layer).ReShape(llama::Shape({kDim, kKVDim})),
          transformer.GetRunState().XB(), V);

      llama::Tensor<float> V_host(V.GetShape(), llama::DeviceType::CPU);
      llama::DeviceFactory::GetDevice(llama::DeviceType::CUDA)
          .GetMemcpy()
          .Copy(*V_host.GetData(), *V.GetData(), V.GetDataBytesSize());

      EXPECT_TRUE(llama::test::tensor_float_compare(
          ref_run_state.v, static_cast<float*>(V_host.GetData()->GetBuffer()),
          kKVDim, 1e-5))
          << "Mismatch in V";
    }
  }
}