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
)

set(TEST_DEP_LIBS
    GTest::gtest_main
    glog
    llama2_cpp_dummy_lib
)

function (add_test_executable target_name)
    add_executable(${target_name} ${ARGN})
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