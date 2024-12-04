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
  cudaDeviceSynchronize();
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
  cudaDeviceSynchronize();
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
  cudaDeviceSynchronize();
}

//------------------------------------------------------------------------------
// End of RoPE
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Start of MultiHeadAttention
//------------------------------------------------------------------------------

__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  float max_val = tid < size ? x[tid] : 0;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, kNumThreadsLarge>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  // exp and sum
  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  // normalize
  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

__global__ void multi_head_attention_kernel(
    const size_t pos, const size_t seq_len, const float* sq,
    const float* key_cache_layer, const float* value_cache_layer,
    const size_t kv_dim, const size_t kv_mul, const size_t head_size,
    float* satt, float* sxb) {
  int h = blockIdx.x;
  // get the query vector for this head
  const float* q = sq + h * head_size;
  // attention scores for this head
  float* att = satt + h * seq_len;
  // iterate over all timesteps, including the current one
  // In CUDA, each thread does a small portion of the calc
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    // get the key vector for this head and at this timestep
    const float* k = key_cache_layer + t * kv_dim + (h / kv_mul) * head_size;
    // calculate the attention score as the dot product of q and k
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
      score += q[i] * k[i];
    }
    score /= sqrtf(head_size);
    // save the score to the attention buffer
    att[t] = score;
  }
  // above was this threads portion of the iteration. wait for all threads to
  // finish
  __syncthreads();

  // softmax the scores to get attention weights, from 0...pos inclusively
  softmax_gpu(att, pos + 1);
  __syncthreads();

  // weighted sum of the values, store back into xb
  // NOTE: by swapping the order of the for loops (vs. C) a simpler
  // version of the code accomplishes the same task and fits more
  // naturally with the CUDA way of subdividing the problem.
  float* xb = sxb + h * head_size;
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      // get the value vector for this head and at this timestep
      const float* v =
          value_cache_layer + t * kv_dim + (h / kv_mul) * head_size;
      // get the attention weight for this timestep
      float a = att[t];
      val += a * v[i];
    }
    xb[i] = val;
  }
}

void LaunchMultiHeadAttentionKernel(
    const size_t pos, const size_t seq_len, const float* sq,
    const float* key_cache_layer, const float* value_cache_layer,
    const size_t kv_dim, const size_t kv_mul, const size_t num_heads,
    const size_t head_size, float* satt, float* sxb) {
  multi_head_attention_kernel<<<num_heads, kNumThreadsLarge>>>(
      pos, seq_len, sq, key_cache_layer, value_cache_layer, kv_dim, kv_mul,
      head_size, satt, sxb);
  cudaDeviceSynchronize();
}

// void multi_head_attention(int pos, Config* p, RunState* s, int kv_dim,
//                           int kv_mul, int head_size, int loff) {
//   multi_head_attention_kernel<<<p->n_heads, num_threads_large>>>(
//       pos, p->max_seq_len, s->q, s->att, s->xb, s->key_cache, s->value_cache,
//       kv_dim, kv_mul, head_size, loff);
// }

//------------------------------------------------------------------------------
// End of MultiHeadAttention
//------------------------------------------------------------------------------

}  // namespace CudaOps

}  // namespace llama
