#include <gtest/gtest.h>

#include <llama.hpp>
#include <random>
#include <references/reference_llama2.cpp>
#include <test_utility.hpp>

class CUDARopeTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}

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

TEST_F(CUDARopeTest, ForwardTest) {
  auto &transformer = llama2_->GetTransformer();
  const auto &tokenizer = llama2_->GetTokenizer();
  const size_t kNumOfLayers = transformer.GetConfig().NumLayers();
  const size_t kDim = transformer.GetConfig().Dim();
  const auto &kWeights = transformer.GetWeights();
  const auto &kConfig = transformer.GetConfig();

  reference_llama2::Transformer ref_transformer;
  reference_llama2::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;
  const auto &kRefConfig = ref_transformer.config;
  const size_t kRefKVDim =
      (kRefConfig.dim * kRefConfig.n_kv_heads) / kRefConfig.n_heads;

  const std::string kPrompt = "Who are you?";
  const size_t kPos = 0;  // First position

  llama::Encoder<float> encoder(tokenizer, kPrompt, true, false,
                                llama::SpecialTokensLlama2());
  auto embed =
      static_cast<float *>(kWeights.TokenEmbeddingTable()->GetBuffer()) +
      kPos * kDim;
  llama::DeviceFactory::GetDevice(kLlamaConfig.device_type)
      .GetMemcpy()
      .Copy(*(llama2_->GetTransformer().GetRunState().X().GetData()),
            (void *)embed, kDim * sizeof(float));

  float *content_row =
      ref_transformer.weights.token_embedding_table + kPos * kDim;
  memcpy(ref_run_state.x, content_row, kDim * sizeof(*ref_run_state.x));

  const size_t kKVDim = (kDim * transformer.GetConfig().NumKVHeads()) /
                        transformer.GetConfig().NumHeads();

  auto &run_state = transformer.GetRunState();
  auto Q = run_state.Q();
  auto X = run_state.X();
  auto XB = run_state.XB();
  auto XB2 = run_state.XB2();
  auto HB = run_state.HB();
  auto HB2 = run_state.HB2();

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    const auto kLayerOffset = layer * kConfig.SeqLen() * kKVDim;
    auto K = run_state.K(layer, kPos).ReShape({kKVDim});
    auto V = run_state.V(layer, kPos).ReShape({kKVDim});

    // RMSNorm
    {
      reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                                ref_weights.rms_att_weight + layer * kDim,
                                kDim);
      op_set_->RmsNorm<float>(transformer.GetRunState().X(),
                              kWeights.RMSAttnWeight(layer), XB);
    }

    // Calculate Q, K, V
    {
      // QKV::Reference
      {
        ref_run_state.k =
            ref_run_state.key_cache + kLayerOffset + kPos * kKVDim;
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
      // QKV::CUDA
      {
        op_set_->MatMul<float>(
            kWeights.WQ(layer).ReShape(llama::Shape({kDim, kDim})), XB, Q);

        op_set_->MatMul<float>(
            kWeights.WK(layer).ReShape(llama::Shape({kDim, kKVDim})), XB, K);

        op_set_->MatMul<float>(
            kWeights.WV(layer).ReShape(llama::Shape({kDim, kKVDim})), XB, V);

        auto Q_host = Q.Dump();
        auto K_host = K.Dump();
        auto V_host = V.Dump();

        EXPECT_TRUE(llama::test::tensor_float_compare(
            ref_run_state.q,
            static_cast<float *>(Q_host.GetData()->GetBuffer()), kDim, 1e-5))
            << "Mismatch in Q at layer " << layer;

        EXPECT_TRUE(llama::test::tensor_float_compare(
            ref_run_state.k,
            static_cast<float *>(K_host.GetData()->GetBuffer()), kKVDim, 1e-5))
            << "Mismatch in K at layer " << layer;

        EXPECT_TRUE(llama::test::tensor_float_compare(
            ref_run_state.v,
            static_cast<float *>(V_host.GetData()->GetBuffer()), kKVDim, 1e-5))
            << "Mismatch in V at layer " << layer;
      }
    }

    // RoPE
    {
      // RoPE::Reference
      {
        const size_t kRefHeadSize = kDim / kRefConfig.n_heads;
        const size_t kNumKVHeads = ref_transformer.config.n_kv_heads;
        const int kRefLayerOffset = layer * kRefConfig.seq_len * kRefKVDim;

        ref_run_state.k =
            ref_run_state.key_cache + kRefLayerOffset + kPos * kRefKVDim;
        ref_run_state.v =
            ref_run_state.value_cache + kRefLayerOffset + kPos * kRefKVDim;
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
      }

      // RoPE::CUDA
      {
        op_set_->RoPE<float>(kPos, transformer.GetConfig(), Q, K);

        llama::Tensor<float> Q_host(Q.GetShape(), llama::DeviceType::CPU);
        llama::DeviceFactory::GetDevice(llama::DeviceType::CUDA)
            .GetMemcpy()
            .Copy(*Q_host.GetData(), *Q.GetData(), Q.GetDataBytesSize());

        llama::Tensor<float> K_host(K.GetShape(), llama::DeviceType::CPU);
        llama::DeviceFactory::GetDevice(llama::DeviceType::CUDA)
            .GetMemcpy()
            .Copy(*K_host.GetData(), *K.GetData(), K.GetDataBytesSize());

        EXPECT_TRUE(llama::test::tensor_float_compare(
            ref_run_state.q,
            static_cast<float *>(Q_host.GetData()->GetBuffer()), kDim, 1e-5))
            << "Mismatch in RoPE(Q) at layer " << layer;

        EXPECT_TRUE(llama::test::tensor_float_compare(
            ref_run_state.k,
            static_cast<float *>(K_host.GetData()->GetBuffer()), kKVDim, 1e-5))
            << "Mismatch in RoPE(K) at layer " << layer;
      }
    }
  }
}