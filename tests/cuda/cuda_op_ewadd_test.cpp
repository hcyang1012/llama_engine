#include <gtest/gtest.h>

#include <device/device.hpp>
#include <dtypes.hpp>
#include <op.hpp>

#include "tensor.hpp"

TEST(CUDA_OP_EWADD_TEST, TEST1) {
  const llama::DeviceType device_type = llama::DeviceType::CUDA;
  llama::Tensor<float> input1({4}, device_type);
  llama::Tensor<float> input2({4}, device_type);
  llama::Tensor<float> output({4}, device_type);

  llama::Tensor<float> input1_cpu({4}, llama::DeviceType::CPU);
  llama::Tensor<float> input2_cpu({4}, llama::DeviceType::CPU);

  for (size_t i = 0; i < 4; i++) {
    input1_cpu[i] = 1.0f;
    input2_cpu[i] = 2.0f;
  }

  llama::DeviceFactory::GetDevice(device_type)
      .GetMemcpy()
      .Copy(*(input1.GetData()), *(input1_cpu.GetData()),
            input1.GetDataBytesSize());
  llama::DeviceFactory::GetDevice(device_type)
      .GetMemcpy()
      .Copy(*(input2.GetData()), *(input2_cpu.GetData()),
            input2.GetDataBytesSize());

  auto op_set_ = llama::CreateOpSet(device_type);
  op_set_->ElementwiseAdd<float>(input1, input2, output);
  auto output_dump = output.Dump();

  for (size_t i = 0; i < 4; i++) {
    EXPECT_FLOAT_EQ(output_dump[i], 3.0f);
  }
}