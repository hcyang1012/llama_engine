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

// CUDA Headers
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// For NVIDIA Transformer Engine
#include <transformer_engine/rmsnorm.h>
#include <transformer_engine/transformer_engine.h>

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
#if 0
  rmsnorm_kernel<<<1, kNumThreadsLarge>>>(x, weight, size, o,
                                          elementsPerThread);
  cudaDeviceSynchronize();
#else
  constexpr size_t kBatch = 1;
  auto build_nvte_tensor = [](const float* dptr, std::vector<size_t> shape,
                              NVTEDType dtype) {
    NVTEShape nvte_shape{shape.data(), shape.size()};
    return nvte_create_tensor(const_cast<float*>(dptr), nvte_shape, dtype,
                              nullptr, nullptr, nullptr);
  };

  auto product = [](const NVTEShape& shape) {
    size_t ret = 1;
    for (size_t i = 0; i < shape.ndim; ++i) {
      ret *= shape.data[i];
    }
    return ret;
  };

  auto build_nvte_tensor2 = [](const float* dptr, NVTEShape shape,
                               NVTEDType dtype) {
    return nvte_create_tensor(const_cast<float*>(dptr), shape, dtype, nullptr,
                              nullptr, nullptr);
  };

  auto get_typesize = [](NVTEDType dtype) {
    switch (dtype) {
      /**
       kNVTEByte = 0,
      kNVTEInt32 = 1,
      kNVTEInt64 = 2,
      kNVTEFloat32 = 3,
      kNVTEFloat16 = 4,
      kNVTEBFloat16 = 5,
      kNVTEFloat8E4M3 = 6,
      kNVTEFloat8E5M2 = 7,
      **/
      case NVTEDType::kNVTEByte:
        return 1;
      case NVTEDType::kNVTEInt32:
        return 4;
      case NVTEDType::kNVTEInt64:
        return 8;
      case NVTEDType::kNVTEFloat32:
        return 4;
      case NVTEDType::kNVTEFloat16:
        return 2;
      case NVTEDType::kNVTEBFloat16:
        return 2;
      default:
        throw std::runtime_error("Unsupported data type");
    }
  };
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  auto input = build_nvte_tensor(x, {kBatch, size}, NVTEDType::kNVTEFloat32);
  auto gamma = build_nvte_tensor(weight, {size}, NVTEDType::kNVTEFloat32);
  auto z = build_nvte_tensor(o, {kBatch, size}, NVTEDType::kNVTEFloat32);
  float* rsigma_dptr = nullptr;
  cudaMalloc(&rsigma_dptr, kBatch * sizeof(float));
  cudaMemset(rsigma_dptr, 0, kBatch * sizeof(float));
  auto rsigma =
      build_nvte_tensor(rsigma_dptr, {kBatch}, NVTEDType::kNVTEFloat32);
  auto workspace = build_nvte_tensor(nullptr, {}, NVTEDType::kNVTEFloat32);
  auto barrier = build_nvte_tensor(nullptr, {}, NVTEDType::kNVTEFloat32);
  constexpr float epsilon = 1e-5;

  // As the workspace and barrier tensors are empty, this function will
  // only set the shape and type of the workspace and barrier tensors to
  // the required values.
  nvte_rmsnorm_fwd(input, gamma, epsilon, z, rsigma, 0,
                   prop.multiProcessorCount, workspace, barrier);
  auto workspace_shape = nvte_tensor_shape(workspace);
  auto workspace_dtype = nvte_tensor_type(workspace);
  float* workspace_dptr;
  cudaMalloc(&workspace_dptr,
             product(workspace_shape) * get_typesize(workspace_dtype));

  workspace =
      build_nvte_tensor2(workspace_dptr, workspace_shape, workspace_dtype);

  auto barrier_shape = nvte_tensor_shape(barrier);
  auto barrier_dtype = nvte_tensor_type(barrier);
  float* barrier_dptr;
  cudaMalloc(&barrier_dptr,
             product(barrier_shape) * get_typesize(barrier_dtype));
  barrier = build_nvte_tensor2(barrier_dptr, barrier_shape, barrier_dtype);

  // Call the function again with the workspace and barrier tensors set to
  // the required values. This time the operation will be performed.
  nvte_rmsnorm_fwd(input, gamma, epsilon, z, rsigma, 0,
                   prop.multiProcessorCount, workspace, barrier);
  cudaDeviceSynchronize();

  nvte_destroy_tensor(input);
  nvte_destroy_tensor(gamma);
  nvte_destroy_tensor(z);
  nvte_destroy_tensor(rsigma);
  nvte_destroy_tensor(workspace);
  nvte_destroy_tensor(barrier);

  cudaFree(rsigma_dptr);
  cudaFree(workspace_dptr);
  cudaFree(barrier_dptr);
#endif
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

//------------------------------------------------------------------------------
// End of MultiHeadAttention
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Start of ElementwiseAdd
//------------------------------------------------------------------------------

__global__ void elementwise_add_kernel(const float* left, const float* right,
                                       float* output, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = left[i] + right[i];
  }
}

void LaunchElementwiseAddKernel(const float* left, const float* right,
                                float* output, int size) {
  elementwise_add_kernel<<<IDivCeil(size, kNumThreadsSmall),
                           kNumThreadsSmall>>>(left, right, output, size);
  cudaDeviceSynchronize();
}

//------------------------------------------------------------------------------
// End of ElementwiseAdd
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Start of SiLU_EWMul
//------------------------------------------------------------------------------
__global__ void f_silu_elementwise_mul_w3_kernel(const float* input,
                                                 const float* weight,
                                                 float* output,
                                                 const size_t size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float val = input[i];
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    val *= (1.0f / (1.0f + expf(-val)));
    // elementwise multiply with w3(x)
    val *= weight[i];
    output[i] = val;
  }
}

void LaunchSiLU_EWMulKernel(const float* input, const float* weight,
                            float* output, const size_t size) {
  f_silu_elementwise_mul_w3_kernel<<<IDivCeil(size, kNumThreadsSmall),
                                     kNumThreadsSmall>>>(input, weight, output,
                                                         size);
  cudaDeviceSynchronize();
}

}  // namespace CudaOps

}  // namespace llama
