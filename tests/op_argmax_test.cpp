#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "op.hpp"
#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif

TEST(OpArgmaxTest, Test1) {
  std::vector<float> input;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < 10; i++) {
    input.push_back(dis(gen));
  }
  llama::Tensor<float> input_tensor({10});
  for (int i = 0; i < 10; i++) {
    input_tensor[i] = input[i];
  }

  size_t expected = reference::sample_argmax(input.data(), 10);
  size_t actual = llama::ArgMax<float>::Compute(input_tensor);

  EXPECT_EQ(expected, actual);
}