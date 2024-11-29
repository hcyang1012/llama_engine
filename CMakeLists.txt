cmake_minimum_required(VERSION 3.0.0)
project(llama2_cpp_dummy VERSION 0.1.0 LANGUAGES C CXX)

include(FetchContent)
FetchContent_Declare(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG v0.4.0
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/download_deps/glog
)
FetchContent_MakeAvailable(glog)


set(CMAKE_CXX_STANDARD 17)
set(LLAMA2_LIB_SOURCES 
    src/config.cpp
    src/transformer.cpp
    src/weights.cpp
)

# add_compile_options(-pg)
# add_link_options(-pg)

# Add Release Build Flags
# Check if ${CMAKE_BUILD_TYPE} is set, if not set to Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    add_compile_options(-O3)
    add_compile_options(-march=native)
else()
    add_compile_options(-O0)
    add_compile_options(-g)
endif()



add_executable(llama2_cpp ${CMAKE_SOURCE_DIR}/llama2.cpp)
target_compile_definitions(llama2_cpp PUBLIC -DUSE_LLAMA2)
target_include_directories(llama2_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(llama2_cpp glog::glog)

# add_executable(llama3_cpp ${CMAKE_SOURCE_DIR}/llama3.cpp)
# target_compile_definitions(llama3_cpp PUBLIC -DUSE_LLAMA3)
# target_include_directories(llama3_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(llama3_cpp glog::glog)


add_executable(llama2_reference ${CMAKE_SOURCE_DIR}/references/reference_llama2.cpp)
target_include_directories(llama2_reference PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(llama2_reference PUBLIC -DINCLUDE_MAIN)

add_executable(llama3_reference ${CMAKE_SOURCE_DIR}/references/reference_llama3.cpp)
target_include_directories(llama3_reference PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(llama3_reference PUBLIC -DINCLUDE_MAIN)



enable_testing()
add_subdirectory(tests)