#include <gtest/gtest.h>

#include <device/malloc/malloc_cuda.hpp>

TEST(CUDA_MALLOC_TEST, BASIC) {
  llama::MemoryAllocatorCUDA cuda_allocator;
  void* ptr;
  const size_t size = 1024;
  auto alloc_result = cuda_allocator.Allocate(&ptr, size);
  EXPECT_EQ(alloc_result, true);
  auto free_result = cuda_allocator.Free(ptr);
  EXPECT_EQ(free_result, true);
}