#include <gtest/gtest.h>

#include "op.hpp"
#include "tensor.hpp"
TEST(OP_EWADD_TEST, TEST1) {
  llama2::Tensor<float> input1({4});
  llama2::Tensor<float> input2({4});
  llama2::Tensor<float> output({4});

  for (size_t i = 0; i < 4; i++) {
    input1[i] = 1.0f;
    input2[i] = 2.0f;
  }

  llama2::ElementwiseAdd<float>::Compute(input1, input2, output);

  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(output[i], 3.0f);
  }
}