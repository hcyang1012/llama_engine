#include <gtest/gtest.h>

#include <string>

#include "reference.hpp"
#include "transformer.hpp"

class LoaderTest : public ::testing::Test {
 public:
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    reference::build_transformer(&transformer_, kCheckPointPath.c_str());
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  reference::Transformer transformer_;

  const std::string kCheckPointPath = "stories15M.bin";
};

TEST_F(LoaderTest, LoadTransformer) {
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
  llama2::Transformer<float> transformer(kCheckPointPath);

  const auto& config = transformer.GetConfig();

  EXPECT_EQ(config.Dim(), transformer_.config.dim);
  EXPECT_EQ(config.HiddenDim(), transformer_.config.hidden_dim);
  EXPECT_EQ(config.NumLayers(), transformer_.config.n_layers);
  EXPECT_EQ(config.NumHeads(), transformer_.config.n_heads);
  EXPECT_EQ(config.NumKVHeads(), transformer_.config.n_kv_heads);
  EXPECT_EQ(config.VocabSize(), transformer_.config.vocab_size);
  EXPECT_EQ(config.SeqLen(), transformer_.config.seq_len);
}