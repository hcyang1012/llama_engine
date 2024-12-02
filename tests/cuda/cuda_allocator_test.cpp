#include <gtest/gtest.h>

#include <device/malloc/malloc_cuda.hpp>

TEST(CUDA_MALLOC_TEST, BASIC) {
  llama::MemoryAllocatorCUDA cuda_allocator;
  void* ptr;
  const size_t size = 1024;
  auto alloc_result = cuda_allocator.Allocate(size);
  EXPECT_NE(alloc_result, nullptr);
}