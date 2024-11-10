#include <gtest/gtest.h>

// For random number generation
#include <random>

#include "op.hpp"
#include "reference.hpp"
TEST(OPTest, rmsnorm_test) {
  const std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> weight = {1.0f, 2.0f, 3.0f, 4.0f};

  std::vector<float> expected(4, 0.0f);
  reference::rmsnorm(expected.data(), x.data(), weight.data(), 4);
  auto actual = llama2::RmsNorm<float>::Compute(x, weight);
}
