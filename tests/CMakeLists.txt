# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://codeload.github.com/google/googletest/zip/refs/tags/v1.15.2
)

FetchContent_MakeAvailable(googletest)

add_executable(loader_test loader_test.cpp)
target_include_directories(loader_test PUBLIC ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_link_libraries(loader_test PRIVATE GTest::gtest_main glog llama2_cpp_dummy_lib)

include(GoogleTest)
gtest_discover_tests(loader_test)