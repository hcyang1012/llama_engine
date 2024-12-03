/**
 * @file op_cuda.cu
 * @brief Library for CUDA operations.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-12-02
 * @note I brought this kernels form llama3.cuda project.
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

//------------------------------------------------------------------------------
// Start of RMSNormq
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// End of RMSNormq
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Start of MatMul
//------------------------------------------------------------------------------

// naive CUDA kernel function to perform matrix multiplication.
// one output per warp so that we can parallelize the dot product across the
// warp Note that ~95% of total time is spent here, so optimizing this is
// important
__global__ void matmul_kernel(const float* weight, const float* input,
                              const int n, const int d, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;

  float sum = 0.0f;
  for (int j = 0; j < n; j++) {
    sum += weight[i * n + j] * input[j];
  }
  out[i] = sum;
}
void LaunchMatMulKernel(const float* weight, const float* input, const int n,
                        const int d, float* out) {
  matmul_kernel<<<IDivCeil(d, kNumThreadsSmall), kNumThreadsSmall>>>(
      weight, input, n, d, out);
}
//------------------------------------------------------------------------------
// End of MatMul
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Start of RoPE
//------------------------------------------------------------------------------

// Additional neural net blocks (brought out from transformer function)
__global__ void rope_kernel(const size_t position, const size_t num_heads,
                            const size_t head_dim, const size_t num_kv_heads,
                            const float freq_scale, float* Q, float* K) {
  int j = threadIdx.x * 2;
  for (int i = 0; i < num_heads; i++) {
    float freq = 1.0f / powf(freq_scale, (float)j / (float)head_dim);
    float val = position * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    const size_t idx = i * head_dim + j;
    float q0 = Q[idx];
    float q1 = Q[idx + 1];
    Q[idx] = q0 * fcr - q1 * fci;
    Q[idx + 1] = q0 * fci + q1 * fcr;
    if (i < num_kv_heads) {
      float k0 = K[idx];
      float k1 = K[idx + 1];
      K[idx] = k0 * fcr - k1 * fci;
      K[idx + 1] = k0 * fci + k1 * fcr;
    }
  }
}

void LaunchRoPEKernel(const size_t position, const size_t num_heads,
                      const size_t head_dim, const size_t num_kv_heads,
                      const float freq_scale, float* Q, float* K) {
  rope_kernel<<<1, head_dim / 2>>>(position, num_heads, head_dim, num_kv_heads,
                                   freq_scale, Q, K);
}

//------------------------------------------------------------------------------
// End of RoPE
//------------------------------------------------------------------------------

}  // namespace CudaOps

}  // namespace llama
