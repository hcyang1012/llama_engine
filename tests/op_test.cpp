#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "op.hpp"
#include "reference.hpp"
#include "transformer.hpp"
#include "weights.hpp"
class RmsNormTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // code here will execute just before the test ensues
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    x_.resize(4);
    weight_.resize(4);
    for (size_t i = 0; i < 4; i++) {
      x_[i] = dis(gen);
      weight_[i] = dis(gen);
    }

    transformer_ = std::make_unique<llama2::Transformer<float>>(kChkPointPath);
  }

  void TearDown() override {
    // code here will be called just after the test completes
    // ok to through exceptions from here if need be
  }

  std::vector<float> x_;
  std::vector<float> weight_;

  const std::string kChkPointPath = "stories15M.bin";

  std::unique_ptr<llama2::Transformer<float>> transformer_;
  std::unique_ptr<llama2::TransformerWeights<float>> weights_;
};

TEST_F(RmsNormTest, RmsNormTest) {
  const size_t kSize = 4;
  std::vector<float> expected_o(kSize);
  auto actual = llama2::RmsNorm<float>::Compute(x_, weight_);
  reference::rmsnorm(expected_o.data(), x_.data(), weight_.data(), kSize);

  EXPECT_EQ(actual.size(), expected_o.size());
}
