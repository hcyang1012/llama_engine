#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "encoder.hpp"
#include "op.hpp"
#include "reference.cpp"
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
    x_ = std::make_unique<llama2::Tensor<float>>(llama2::Shape({kSize}));
    weight_ = std::make_unique<llama2::Tensor<float>>(llama2::Shape({kSize}));

    for (size_t i = 0; i < kSize; i++) {
      (*x_)[i] = dis(gen);
      (*weight_)[i] = dis(gen);
    }

    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
    tokenizer_ = std::make_unique<llama2::Tokenizer<float>>(
        kTokenizerBinPath, transformer_->GetConfig().VocabSize());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  std::unique_ptr<llama2::Tensor<float>> x_;
  std::unique_ptr<llama2::Tensor<float>> weight_;

  const std::string kChkPointPath = "stories15M.bin";
  const std::string kTokenizerBinPath = "tokenizer.bin";

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::Tokenizer<float>> tokenizer_;
};

TEST_F(AttentionTest, ForwardTest) {
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
            transformer_->GetRunState().X()->GetData());

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
          transformer_->GetRunState().XB()->GetData(),
          transformer_->GetRunState().X()->GetData(),
          kWeights.RMSAttnWeight() + layer * kDim, kDim);

      EXPECT_TRUE(std::equal(ref_run_state.xb, ref_run_state.xb + kDim,
                             transformer_->GetRunState().XB()->GetData()));
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

      transformer_->GetRunState().UpdateKV(layer, kPos);
      llama2::MatMul<float>::Compute(
          transformer_->GetRunState().Q()->GetData(),
          transformer_->GetRunState().XB()->GetData(),
          kWeights.WQ() + layer * kDim * kDim, kDim, kDim);

      llama2::MatMul<float>::Compute(
          transformer_->GetRunState().K(),
          transformer_->GetRunState().XB()->GetData(),
          kWeights.WK() + layer * kDim * kKVDim, kDim, kKVDim);

      llama2::MatMul<float>::Compute(
          transformer_->GetRunState().V(),
          transformer_->GetRunState().XB()->GetData(),
          kWeights.WV() + layer * kDim * kKVDim, kDim, kKVDim);

      EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                             transformer_->GetRunState().Q()->GetData()));
      EXPECT_TRUE(std::equal(ref_run_state.k, ref_run_state.k + kRefKVDim,
                             transformer_->GetRunState().K()));
      EXPECT_TRUE(std::equal(ref_run_state.v, ref_run_state.v + kRefKVDim,
                             transformer_->GetRunState().V()));
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

      llama2::RoPE::Compute(*(transformer_->GetRunState().Q()), kPos,
                            transformer_->GetConfig(),
                            *(transformer_->GetRunState().Q()), true);

      EXPECT_TRUE(std::equal(ref_run_state.q, ref_run_state.q + kDim,
                             transformer_->GetRunState().Q()->GetData()));
    }

    // Attention
    {
      const size_t kPerHeadDim = kDim / transformer_->GetConfig().NumHeads();
      for (size_t header_idx = 0; header_idx < kRefConfig.n_heads;
           ++header_idx) {
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

        reference::softmax(att, kPos + 1);

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

        auto Q =
            llama2::Tensor<float>{transformer_->GetRunState().Q()->GetData(),
                                  llama2::Shape({kPerHeadDim})};
        auto K = llama2::Tensor<float>{
            transformer_->GetRunState().K(),
            llama2::Shape({kKVDim, static_cast<size_t>(
                                       transformer_->GetConfig().SeqLen())})};
        auto V = llama2::Tensor<float>{
            transformer_->GetRunState().V(),
            llama2::Shape({kKVDim, static_cast<size_t>(
                                       transformer_->GetConfig().SeqLen())})};
        auto output =
            llama2::Tensor<float>{transformer_->GetRunState().XB()->GetData() +
                                      header_idx * kPerHeadDim,
                                  llama2::Shape({kPerHeadDim})};
        llama2::Attention<float>::Compute(Q, K, V, transformer_->GetConfig(),
                                          kPos, header_idx, output);

        // Compare the output
        EXPECT_TRUE(std::equal(xb, xb + kRefHeadSize, output.GetData()));
      }
    }
  }
}