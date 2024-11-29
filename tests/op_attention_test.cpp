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
class AttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    const size_t kSize = 4;
    x_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                op_set_->GetDeviceType());
    weight_ = std::make_unique<llama::Tensor<float>>(llama::Shape({kSize}),
                                                     op_set_->GetDeviceType());

    for (size_t i = 0; i < kSize; i++) {
      (*x_)[i] = dis(gen);
      (*weight_)[i] = dis(gen);
    }

    transformer_ = std::make_unique<llama::Transformer<float>>(
        kChkPointPath, *op_set_, llama::SpecialTokensLlama2());
    tokenizer_ = std::make_unique<llama::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  std::unique_ptr<llama::Tensor<float>> x_;
  std::unique_ptr<llama::Tensor<float>> weight_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama::Transformer<float>> transformer_;
  std::unique_ptr<llama::Tokenizer<float>> tokenizer_;

  std::unique_ptr<llama::OpSet> op_set_ =
      llama::CreateOpSet(llama::DeviceType::CPU);
};

TEST_F(AttentionTest, ForwardTest) {
  const size_t kNumOfLayers = transformer_->GetConfig().NumLayers();
  const size_t kDim = transformer_->GetConfig().Dim();
  const auto &kWeights = transformer_->GetWeights();

  reference_llama2::Transformer ref_transformer;
  reference_llama2::build_transformer(&ref_transformer, kChkPointPath.c_str());
  const auto ref_weights = ref_transformer.weights;
  auto ref_run_state = ref_transformer.state;

  const std::string kPrompt = "Who are you?";

  const size_t kPos = 0;  // First position
  llama::Encoder<float> encoder(*tokenizer_, kPrompt, true, false,
                                llama::SpecialTokensLlama2());
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
      reference_llama2::rmsnorm(ref_run_state.xb, ref_run_state.x,
                                ref_weights.rms_att_weight + layer * kDim,
                                kDim);

      op_set_->RmsNorm<float>(transformer_->GetRunState().X(),
                              kWeights.RMSAttnWeight(layer),
                              transformer_->GetRunState().XB());

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
          transformer_->GetRunState().XB(), transformer_->GetRunState().Q());

      auto K = transformer_->GetRunState().K(layer, kPos).ReShape({kKVDim});
      op_set_->MatMul<float>(
          kWeights.WK(layer).ReShape(llama::Shape({kDim, kRefKVDim})),
          transformer_->GetRunState().XB(), K);

      auto V = transformer_->GetRunState().V(layer, kPos).ReShape({kKVDim});
      op_set_->MatMul<float>(
          kWeights.WV(layer).ReShape(llama::Shape({kDim, kRefKVDim})),
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
      auto K = transformer_->GetRunState().K(layer, kPos);
      op_set_->RoPE<float>(kPos, transformer_->GetConfig(), Q, K);

      EXPECT_TRUE(
          std::equal(ref_run_state.q, ref_run_state.q + kDim, Q.GetData()));
    }

    // Attention
    {
      const size_t kPerHeadDim = kDim / transformer_->GetConfig().NumHeads();
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

        auto Q = transformer_->GetRunState().Q(header_idx);
        auto K = transformer_->GetRunState().K(layer);
        auto V = transformer_->GetRunState().V(layer);

        auto output = transformer_->GetRunState().XB(header_idx);
        op_set_->Attention<float>(Q, K, V, transformer_->GetConfig(), kPos,
                                  kKVHeadIdx, output);

        EXPECT_TRUE(std::equal(xb, xb + kRefHeadSize, output.GetData()))
            << "Compare for header #" << header_idx << " failed.";
      }
    }
  }
}