
# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://codeload.github.com/google/googletest/zip/refs/tags/v1.15.2
)

FetchContent_MakeAvailable(googletest)
include(GoogleTest)

set(TEST_INCLUDE_PATH
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tests
)

set(TEST_DEP_LIBS
    GTest::gtest_main
    glog
)

function (add_test_executable target_name)
    add_executable(${target_name} ${ARGN})
    target_compile_definitions(${target_name} PUBLIC -DUSE_LLAMA2)
    target_include_directories(${target_name} PUBLIC ${TEST_INCLUDE_PATH})
    target_link_libraries(${target_name} PRIVATE ${TEST_DEP_LIBS})
    gtest_discover_tests(${target_name})
endfunction()

add_test_executable(loader_test loader_test.cpp)
add_test_executable(tokenizer_test tokenizer_test.cpp)
add_test_executable(sampler_test sampler_test.cpp)
add_test_executable(encode_test encode_test.cpp)
add_test_executable(op_rmsnorm_test op_rmsnorm_test.cpp)
add_test_executable(op_mm_test op_mm_test.cpp)
add_test_executable(op_rope_test op_rope_test.cpp)
add_test_executable(op_softmax_test op_softmax_test.cpp)
add_test_executable(op_attention_test op_attention_test.cpp)
add_test_executable(op_ewadd_test op_ewadd_test.cpp)
add_test_executable(forward_test forward_test.cpp)
add_test_executable(op_argmax_test op_argmax_test.cpp)
add_test_executable(generate_test generate_test.cpp)

function (add_cuda_test target_name)
    add_test_executable(${target_name} ${ARGN})
    append_cuda_deps(${target_name})
    target_link_libraries(${target_name} PRIVATE op_cuda)
endfunction()

# CUDA tests
if(USE_CUDA)

    build_cuda_file(op_cuda ${CMAKE_SOURCE_DIR}/include/op_cuda.cu)

    add_cuda_test(cuda_allocator_test cuda/cuda_allocator_test.cpp)
    add_cuda_test(cuda_op_rmsnorm_test cuda/cuda_op_rmsnorm_test.cpp)
    add_cuda_test(cuda_op_mm_test cuda/cuda_op_mm_test.cpp)
    add_cuda_test(cuda_op_rope_test cuda/cuda_op_rope_test.cpp)
    add_cuda_test(cuda_op_attention_test cuda/cuda_op_attention_test.cpp)
    add_cuda_test(cuda_op_ewadd_test cuda/cuda_op_ewadd_test.cpp)
    add_cuda_test(cuda_forward_test cuda/cuda_forward_test.cpp)
    add_cuda_test(cuda_generate_test cuda/cuda_generate_test.cpp)
endif()
