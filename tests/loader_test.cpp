#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>

#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif
#include "transformer.hpp"
class LoaderTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    reference_llama2::build_transformer(&ref_transformer_,
                                        kCheckPointPath.c_str());
    transformer_ =
        std::make_unique<llama::Transformer<float>>(kCheckPointPath, *op_set_);
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  reference_llama2::Transformer ref_transformer_;
  std::unique_ptr<llama::Transformer<float>> transformer_;

  const llama::DeviceType kDeviceType = llama::DeviceType::CPU;
  std::unique_ptr<llama::OpSet> op_set_ = llama::CreateOpSet(kDeviceType);

  const std::string kCheckPointPath = "stories15M.bin";
};

class ConfigLoadTest : public LoaderTest {
 public:
 protected:
  void SetUp() override { LoaderTest::SetUp(); }

  void TearDown() override { LoaderTest::TearDown(); }

  const llama::DeviceType kDeviceType = kDeviceType;
};

TEST_F(ConfigLoadTest, ElementWiseCheck) {
  /**
int dim;         // transformer dimension
int hidden_dim;  // for ffn layers
int n_layers;    // number of layers
int n_heads;     // number of query heads
int n_kv_heads;  // number of key/value heads (can be < query heads because of
                 // multiquery)
int vocab_size;  // vocabulary size, usually 256 (byte-level)
int seq_len;     // max sequence length
   */
  auto op_set = llama::CreateOpSet(kDeviceType);
  llama::Transformer<float> transformer(kCheckPointPath, *op_set);

  const auto& config = transformer_->GetConfig();

  EXPECT_EQ(config.Dim(), ref_transformer_.config.dim);
  EXPECT_EQ(config.HiddenDim(), ref_transformer_.config.hidden_dim);
  EXPECT_EQ(config.NumLayers(), ref_transformer_.config.n_layers);
  EXPECT_EQ(config.NumHeads(), ref_transformer_.config.n_heads);
  EXPECT_EQ(config.NumKVHeads(), ref_transformer_.config.n_kv_heads);
  EXPECT_EQ(config.VocabSize(), ref_transformer_.config.vocab_size);
  EXPECT_EQ(config.SeqLen(), ref_transformer_.config.seq_len);
}

class WeightLoadTest : public LoaderTest {
 public:
 protected:
  void SetUp() override {
    LoaderTest::SetUp();
    kHeadSize =
        transformer_->GetConfig().Dim() / transformer_->GetConfig().NumHeads();
  }

  void TearDown() override { LoaderTest::TearDown(); }

  size_t kHeadSize = 0;
};

TEST_F(WeightLoadTest, TokenEmbeddingTableTest) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kTokenEmbeddingTableSize = kConfig.VocabSize() * kConfig.Dim();
  const auto p_tok_emb_table = kWeights.TokenEmbeddingTable();
  // Contents of transformer_.weights.token_embedding_table should be equal to
  // contents of kWeights.token_embedding_table
  EXPECT_TRUE(std::equal(p_tok_emb_table,
                         p_tok_emb_table + kTokenEmbeddingTableSize,
                         ref_transformer_.weights.token_embedding_table));
}

TEST_F(WeightLoadTest, RMSAttWeightTest) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kRMSAttWeightSize = kConfig.NumLayers() * kConfig.Dim();
  const auto p_rms_att_weight = kWeights.RMSAttnWeight();
  // Contents of transformer_.weights.rms_att_weight should be equal to
  // contents of kWeights.rms_att_weight
  EXPECT_TRUE(std::equal(p_rms_att_weight, p_rms_att_weight + kRMSAttWeightSize,
                         ref_transformer_.weights.rms_att_weight));
}

TEST_F(WeightLoadTest, WQTest) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kWQSize =
      kConfig.NumLayers() * kConfig.Dim() * (kConfig.NumHeads() * kHeadSize);
  const auto p_wq = kWeights.WQ();
  // Contents of transformer_.weights.wq should be equal to
  // contents of kWeights.wq
  EXPECT_TRUE(std::equal(p_wq, p_wq + kWQSize, ref_transformer_.weights.wq));
}

TEST_F(WeightLoadTest, WKTest) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kWKSize =
      kConfig.NumLayers() * kConfig.Dim() * (kConfig.NumKVHeads() * kHeadSize);
  const auto p_wk = kWeights.WK();
  // Contents of transformer_.weights.wk should be equal to
  // contents of kWeights.wk
  EXPECT_TRUE(std::equal(p_wk, p_wk + kWKSize, ref_transformer_.weights.wk));
}

TEST_F(WeightLoadTest, WVTest) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kWVSize =
      kConfig.NumLayers() * kConfig.Dim() * (kConfig.NumKVHeads() * kHeadSize);
  const auto p_wv = kWeights.WV();
  // Contents of transformer_.weights.wv should be equal to
  // contents of kWeights.wv
  EXPECT_TRUE(std::equal(p_wv, p_wv + kWVSize, ref_transformer_.weights.wv));
}

TEST_F(WeightLoadTest, WO) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kWOSize =
      kConfig.NumLayers() * (kConfig.NumHeads() * kHeadSize) * kConfig.Dim();
  const auto p_wo = kWeights.WO();
  // Contents of transformer_.weights.wo should be equal to
  // contents of kWeights.wo
  EXPECT_TRUE(std::equal(p_wo, p_wo + kWOSize, ref_transformer_.weights.wo));
}

TEST_F(WeightLoadTest, RMSFFNWeight) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kRMSFFNWeightSize = kConfig.NumLayers() * kConfig.Dim();
  const auto p_rms_ffn_weight = kWeights.RMSFFNWeight();
  // Contents of transformer_.weights.rms_ffn_weight should be equal to
  // contents of kWeights.rms_ffn_weight
  EXPECT_TRUE(std::equal(p_rms_ffn_weight, p_rms_ffn_weight + kRMSFFNWeightSize,
                         ref_transformer_.weights.rms_ffn_weight));
}

TEST_F(WeightLoadTest, W1) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kW1Size =
      kConfig.NumLayers() * kConfig.HiddenDim() * kConfig.Dim();
  const auto p_w1 = kWeights.W1();
  // Contents of transformer_.weights.w1 should be equal to
  // contents of kWeights.w1
  EXPECT_TRUE(std::equal(p_w1, p_w1 + kW1Size, ref_transformer_.weights.w1));
}

TEST_F(WeightLoadTest, W2) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kW2Size =
      kConfig.NumLayers() * kConfig.Dim() * kConfig.HiddenDim();
  const auto p_w2 = kWeights.W2();
  // Contents of transformer_.weights.w2 should be equal to
  // contents of kWeights.w2
  EXPECT_TRUE(std::equal(p_w2, p_w2 + kW2Size, ref_transformer_.weights.w2));
}

TEST_F(WeightLoadTest, W3) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kW3Size =
      kConfig.NumLayers() * kConfig.HiddenDim() * kConfig.Dim();
  const auto p_w3 = kWeights.W3();
  // Contents of transformer_.weights.w3 should be equal to
  // contents of kWeights.w3
  EXPECT_TRUE(std::equal(p_w3, p_w3 + kW3Size, ref_transformer_.weights.w3));
}

TEST_F(WeightLoadTest, RMSFinalWeight) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kRMSFinalWeightSize = kConfig.Dim() +
                                     (kConfig.SeqLen() * kHeadSize / 2) +
                                     (kConfig.SeqLen() * kHeadSize / 2);

  const auto p_rms_final_weight = kWeights.RMSFinalWeight();

  // Contents of transformer_.weights.rms_final_weight should be equal to
  // contents of kWeights.rms_final_weight
  EXPECT_TRUE(std::equal(p_rms_final_weight.GetData(),
                         p_rms_final_weight.GetData() + kRMSFinalWeightSize,
                         ref_transformer_.weights.rms_final_weight));
}

TEST_F(WeightLoadTest, WCLS) {
  const auto& kConfig = transformer_->GetConfig();

  const auto& kWeights = transformer_->GetWeights();
  const size_t kWCLSSize = kConfig.Dim() * kConfig.VocabSize();
  const auto p_wcls = kWeights.WCLS();
  // Contents of transformer_.weights.wcls should be equal to
  // contents of kWeights.wcls
  EXPECT_TRUE(std::equal(p_wcls.GetData(), p_wcls.GetData() + kWCLSSize,
                         ref_transformer_.weights.wcls));

  LOG(WARNING) << "The size of WSL is not checked yet.";
}

class RunStateAllocTest : public LoaderTest {
 public:
 protected:
  void SetUp() override { LoaderTest::SetUp(); }

  void TearDown() override { LoaderTest::TearDown(); }
  const llama::DeviceType kDeviceType = kDeviceType;
};

TEST_F(RunStateAllocTest, AllocSizeTest) {
  const auto& kConfig = transformer_->GetConfig();
  llama::RunState<float> run_state(kConfig, kDeviceType);

  const size_t kDim = kConfig.Dim();
  const size_t kHiddenDim = kConfig.HiddenDim();
  const size_t kNumLayers = kConfig.NumLayers();
  const size_t kNumHeads = kConfig.NumHeads();
  const size_t kNumKVHeads = kConfig.NumKVHeads();
  const size_t kVocabSize = kConfig.VocabSize();
  const size_t kSeqLen = kConfig.SeqLen();

  const size_t kKVDims = (kDim * kNumKVHeads) / kNumHeads;

  const size_t kXSize = kDim;
  const size_t kXBSize = kDim;
  const size_t kXB2Size = kDim;
  const size_t kHBSize = kHiddenDim;
  const size_t kHB2Size = kHiddenDim;
  const size_t kQSize = kDim;
  const size_t kKeyCacheSize = kNumLayers * kSeqLen * kKVDims;
  const size_t kValueCacheSize = kNumLayers * kSeqLen * kKVDims;
  const size_t kAttSize = kNumHeads * kSeqLen;
  const size_t kLogitsSize = kVocabSize;

  EXPECT_EQ(run_state.X().GetShape().GetSize(), kXSize);
  EXPECT_EQ(run_state.XB().GetShape().GetSize(), kXBSize);
  EXPECT_EQ(run_state.XB2().GetShape().GetSize(), kXB2Size);
  EXPECT_EQ(run_state.HB().GetShape().GetSize(), kHBSize);
  EXPECT_EQ(run_state.HB2().GetShape().GetSize(), kHB2Size);
  EXPECT_EQ(run_state.Q().GetShape().GetSize(), kQSize);
  EXPECT_EQ(run_state.KeyCache().GetShape().GetSize(), kKeyCacheSize);
  EXPECT_EQ(run_state.ValueCache().GetShape().GetSize(), kValueCacheSize);
  EXPECT_EQ(run_state.Att().GetShape().GetSize(), kAttSize);
  EXPECT_EQ(run_state.Logits().GetShape().GetSize(), kLogitsSize);
}