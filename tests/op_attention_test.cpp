#include <gtest/gtest.h>

#include <llama.hpp>
#include <memory>
#include <random>
#include <references/reference_llama2.cpp>
class AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    llama::LlamaConfig llama_config = {kChkPointPath, kTokenizerBinPath,
                                       llama::DeviceType::CPU};
    llama2_ = std::make_unique<llama::Llama2<float>>(llama_config);
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t kSize = 4;
    x_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                llama_config.device_type);
    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                     llama_config.device_type);

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

TEST_F(AttentionTest, ForwardTest) {
  auto &transformer = llama2_->GetTransformer();
  const auto &tokenizer = llama2_->GetTokenizer();
  const size_t kNumOfLayers = transformer.GetConfig().NumLayers();
  const size_t kDim = transformer.GetConfig().Dim();
  const auto &kWeights = transformer.GetWeights();

  reference_llama2::Transformer ref_transformer;
  reference_llama2::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama::Encoder<float> encoder(tokenizer, kPrompt, true, false,
                                llama::SpecialTokensLlama2());
  auto content_row =
      static_cast<float *>(kWeights.TokenEmbeddingTable()->GetBuffer()) +
      kPos * kDim;
  llama::GetMemcpy(kLlamaConfig.device_type)
      .Copy(*(transformer.GetRunState().X().GetData()), (void *)content_row,
            kDim * sizeof(float));
  // std::copy(content_row, content_row + kDim,
  //           transformer.GetRunState().X().GetData());

  std::copy(content_row, content_row + kDim, ref_run_state.x);

  const auto &kRefConfig = ref_transformer.config;
  const size_t kRefKVDim =
      (kRefConfig.dim * kRefConfig.n_kv_heads) / kRefConfig.n_heads;

  const size_t kKVDim = (kDim * transformer.GetConfig().NumKVHeads()) /
                        transformer.GetConfig().NumHeads();

  for (size_t layer = 0; layer < kNumOfLayers; layer++) {
    // RMSNorm
    {
      reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                                ref_weights.rms_att_weight + layer * kDim,
                                kDim);

      op_set_->RmsNorm<float>(transformer.GetRunState().X(),
                              kWeights.RMSAttnWeight(layer),
                              transformer.GetRunState().XB());

      EXPECT_TRUE(std::equal(
          ref_run_state.xb, ref_run_state.xb + kDim,
          static_cast<float *>(
              transformer.GetRunState().XB().GetData()->GetBuffer())));
    }

    const int kRefLayerOffset = layer * kRefConfig.seq_len * kRefKVDim;
    ref_run_state.k =
        ref_run_state.key_cache + kRefLayerOffset + kPos * kRefKVDim;
    ref_run_state.v =
        ref_run_state.value_cache + kRefLayerOffset + kPos * kRefKVDim;

    // Calculate QKV
    {
      // qkv matmuls for this position
      reference_llama2::matmul(ref_run_state.q, ref_run_state.xb,
                               ref_weights.wq + layer * kDim * kDim, kDim,
                               kDim);
      reference_llama2::matmul(ref_run_state.k, ref_run_state.xb,
                               ref_weights.wk + layer * kDim * kRefKVDim, kDim,
                               kRefKVDim);
      reference_llama2::matmul(ref_run_state.v, ref_run_state.xb,
                               ref_weights.wv + layer * kDim * kRefKVDim, kDim,
                               kRefKVDim);

      op_set_->MatMul<float>(
          kWeights.WQ(layer).ReShape(llama::Shape({kDim, kDim})),
          transformer.GetRunState().XB(), transformer.GetRunState().Q());

      auto K = transformer.GetRunState().K(layer, kPos).ReShape({kKVDim});
      op_set_->MatMul<float>(
          kWeights.WK(layer).ReShape(llama::Shape({kDim, kRefKVDim})),
          transformer.GetRunState().XB(), K);

      auto V = transformer.GetRunState().V(layer, kPos).ReShape({kKVDim});
      op_set_->MatMul<float>(
          kWeights.WV(layer).ReShape(llama::Shape({kDim, kRefKVDim})),
          transformer.GetRunState().XB(), V);

      EXPECT_TRUE(std::equal(
          ref_run_state.q, ref_run_state.q + kDim,
          static_cast<float *>(
              transformer.GetRunState().Q().GetData()->GetBuffer())));
      EXPECT_TRUE(std::equal(ref_run_state.k, ref_run_state.k + kRefKVDim,
                             static_cast<float *>(transformer.GetRunState()
                                                      .K(layer, kPos)
                                                      .GetData()
                                                      ->GetBuffer())));
      EXPECT_TRUE(std::equal(ref_run_state.v, ref_run_state.v + kRefKVDim,
                             static_cast<float *>(transformer.GetRunState()
                                                      .V(layer, kPos)
                                                      .GetData()
                                                      ->GetBuffer())));
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
      auto Q = transformer.GetRunState().Q();
      auto K = transformer.GetRunState().K(layer, kPos);
      op_set_->RoPE<float>(kPos, transformer.GetConfig(), Q, K);

      EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                             static_cast<float *>(Q.GetData()->GetBuffer())));
    }

    // Attention
    {
      const size_t kPerHeadDim = kDim / transformer.GetConfig().NumHeads();
      for (size_t header_idx = 0; header_idx < kRefConfig.n_heads;
           ++header_idx) {
        const size_t kKVHeadIdx = header_idx / kRefConfig.n_kv_heads;
        float *q = ref_run_state.q + header_idx * kRefHeadSize;
        float *att = ref_run_state.att + header_idx * kRefConfig.seq_len;

        for (int t = 0; t <= kPos; t++) {
          float *k = ref_run_state.key_cache + kRefLayerOffset + t * kRefKVDim +
                     (header_idx / kRefKVDim) * kRefHeadSize;
          float score = 0.0f;
          for (int i = 0; i < kRefHeadSize; i++) {
            score += q[i] * k[i];
          }
          score /= sqrtf(kRefHeadSize);
          att[t] = score;
        }

        reference_llama2::softmax(att, kPos + 1);

        float *xb = ref_run_state.xb + header_idx * kRefHeadSize;
        memset(xb, 0, kRefHeadSize * sizeof(float));
        for (int t = 0; t <= kPos; t++) {
          float *v = ref_run_state.value_cache + kRefLayerOffset +
                     t * kRefKVDim + (header_idx / kRefKVDim) * kRefHeadSize;
          float a = att[t];
          for (int i = 0; i < kRefHeadSize; i++) {
            xb[i] += a * v[i];
          }
        }

        auto Q = transformer.GetRunState().Q(header_idx);
        auto K = transformer.GetRunState().K(layer);
        auto V = transformer.GetRunState().V(layer);

        auto output = transformer.GetRunState().XB(header_idx);
        op_set_->Attention<float>(Q, K, V, transformer.GetConfig(), kPos,
                                  kKVHeadIdx, output);

        EXPECT_TRUE(
            std::equal(xb, xb + kRefHeadSize,
                       static_cast<float *>(output.GetData()->GetBuffer())))
            << "Compare for header #" << header_idx << " failed.";
      }
    }
  }
}