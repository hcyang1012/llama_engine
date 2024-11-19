#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "encoder.hpp"
#include "op.hpp"
#include "reference.cpp"
#include "transformer.hpp"
class RopeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
  }

  void reference_rope(const float* input, const int position, float* output) {
    const auto dim = transformer_->GetConfig().Dim();
    const auto kv_dim = (transformer_->GetConfig().Dim() *
                         transformer_->GetConfig().NumKVHeads()) /
                        transformer_->GetConfig().NumHeads();
    const auto head_size = dim / transformer_->GetConfig().NumHeads();
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float theta = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = position * theta;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int v = 0; v < rotn; v++) {
        float v0 = input[i];
        float v1 = input[i + 1];
        output[i] = v0 * fcr - v1 * fci;
        output[i + 1] = v0 * fci + v1 * fcr;
      }
    }
  }

  void TearDown() override {}

  const std::string kChkPointPath = "stories15M.bin";

  std::unique_ptr<llama2::Transformer<float>> transformer_;
};

TEST_F(RopeTest, Test) {
  llama2::Tensor<float> input(
      {static_cast<size_t>(transformer_->GetConfig().Dim())});
  llama2::Tensor<float> output(
      {static_cast<size_t>(transformer_->GetConfig().Dim())});

  const int position = 0;

  // Fill the input tensor with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < input.GetShape()[0]; i++) {
    input[i] = dis(gen);
  }

  // Compute the output tensor using the RoPE operation
  llama2::RoPE::Compute(input, position, transformer_->GetConfig(), output);
  // Reference implementation
  std::vector<float> ref_output(transformer_->GetConfig().Dim());
  reference_rope(input.GetData(), position, ref_output.data());

  // Compare the output tensors
  for (size_t i = 0; i < output.GetShape()[0]; i++) {
    EXPECT_FLOAT_EQ(output[i], ref_output[i]);
  }
}

TEST_F(RopeTest, OverWriteTest) {
  llama2::Tensor<float> input(
      {static_cast<size_t>(transformer_->GetConfig().Dim())});
  llama2::Tensor<float> output(
      {static_cast<size_t>(transformer_->GetConfig().Dim())});

  const int position = 0;

  // Fill the input tensor with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < input.GetShape()[0]; i++) {
    input[i] = dis(gen);
  }

  // Fill the output tensor with random values
  for (size_t i = 0; i < output.GetShape()[0]; i++) {
    output[i] = dis(gen);
  }

  // Compute the output tensor using the RoPE operation
  llama2::RoPE::Compute(input, position, transformer_->GetConfig(), output,
                        true);
  // Reference implementation
  std::vector<float> ref_output(transformer_->GetConfig().Dim());
  reference_rope(input.GetData(), position, ref_output.data());

  // Compare the output tensors
  for (size_t i = 0; i < output.GetShape()[0]; i++) {
    EXPECT_FLOAT_EQ(output[i], ref_output[i]);
  }
}