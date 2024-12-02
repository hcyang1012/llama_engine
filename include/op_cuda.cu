/**
 * @file op_cuda.cu
 * @brief Library for CUDA operations.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-02
 * @note I bought this kernels form llama3.cuda project.
 *       (https://github.com/likejazz/llama3.cuda)
 *       Thanks @likejazz
 */

// C System-Headers

// C++ System-Headers
#include <cstddef>  // For size_t

// Project Headers

// Third-party Headers
#include <cublas_v2.h>

#include <cub/cub.cuh>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace llama {
namespace CudaOps {

inline size_t IDivCeil(size_t a, size_t b) { return (a + b - 1) / b; }

constexpr size_t kNumThreadsLarge = 1024;
constexpr size_t kNumThreadsSmall = 64;

__global__ void rmsnorm_kernel(const float* x, const float* weight,
                               const size_t size, float* o,
                               const size_t elementsPerThread) {
  // parallel reduction of sum of squares via CUB
  float ss = 0.0f;
  for (int i = 0; i < elementsPerThread; i++) {
    int j = threadIdx.x + i * kNumThreadsLarge;
    if (j < size) ss += x[j] * x[j];
  }
  using BlockReduce = cub::BlockReduce<float, kNumThreadsLarge>;
  __shared__ typename BlockReduce::TempStorage temp;
  ss = BlockReduce(temp).Sum(ss);

  // serialization point to calculate normalization factor
  __shared__ float shared_ss;
  if (threadIdx.x == 0) {
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    shared_ss = ss;
  }
  __syncthreads();
  ss = shared_ss;

  // normalize and scale
  for (int i = 0; i < elementsPerThread; i++) {
    int j = threadIdx.x + i * kNumThreadsLarge;
    if (j < size) {
      o[j] = weight[j] * (ss * x[j]);
    }
  }
}

void LaunchRmsNormKernel(const float* x, const float* weight, size_t size,
                         float* o, cudaStream_t stream = nullptr) {
  const auto elementsPerThread = IDivCeil(size, kNumThreadsLarge);
  rmsnorm_kernel<<<1, kNumThreadsLarge>>>(x, weight, size, o,
                                          elementsPerThread);
}

}  // namespace CudaOps

}  // namespace llama
