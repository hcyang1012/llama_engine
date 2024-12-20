#include <gtest/gtest.h>

#include <op.hpp>
#include <random>
#include <vector>
#if defined(USE_LLAMA2)
#include "references/reference_llama2.cpp"
#endif

TEST(OpArgmaxTest, Test1) {
  const llama::DeviceType device_type = llama::DeviceType::CPU;
  std::vector<float> input;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (int i = 0; i < 10; i++) {
    input.push_back(dis(gen));
  }
  llama::Tensor<float> input_tensor({10}, device_type);
  for (int i = 0; i < 10; i++) {
    input_tensor[i] = input[i];
  }

  size_t expected = reference_llama2::sample_argmax(input.data(), 10);

  auto op_set = llama::CreateOpSet(device_type);
  auto actual = op_set->ArgMax<float>(input_tensor);

  EXPECT_EQ(expected, actual);
}