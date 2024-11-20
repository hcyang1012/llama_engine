#include <gtest/gtest.h>

#include <random>

#include "op.hpp"
#include "reference.cpp"
#include "tensor.hpp"

TEST(SoftMaxTest, BATCH_TEST) {
  const size_t kMinDim = 1;
  const size_t kMaxDim = 1024;
  const size_t kMinBatch = 1;
  const size_t kMaxBatch = 5;
  const size_t kRepeat = 20;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis_dim(kMinDim, kMaxDim);
  std::uniform_int_distribution<size_t> dis_batch(kMinBatch, kMaxBatch);

  for (size_t iter = 0; iter < kRepeat; ++iter) {
    // Select a random dimension
    const size_t dim = dis_dim(gen);
    const size_t batch = dis_batch(gen);
    llama2::Tensor<float> input({batch, dim});
    llama2::Tensor<float> output({batch, dim});

    std::vector<float> reference_inout(batch * dim);

    for (size_t i = 0; i < batch * dim; i++) {
      reference_inout[i] = static_cast<float>(i);
      input[i] = static_cast<float>(i);
    }

    llama2::SoftMax<float>::Compute(input, output);
    for (size_t b = 0; b < batch; b++) {
      reference::softmax(reference_inout.data() + b * dim, dim);
      for (size_t i = 0; i < dim; i++) {
        EXPECT_NEAR(output[b * dim + i], reference_inout[b * dim + i], 1e-5);
      }
    }
  }
}