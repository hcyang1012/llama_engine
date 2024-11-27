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

# Add Build options to select the version of llama: LLAMA2 or LLAMA3
option(LLAMA_VERSION "Select the version of LLAMA to use" "LLAMA2")
if (${LLAMA_VERSION} STREQUAL "LLAMA2")
    add_compile_definitions(USE_LLAMA2)
elseif (${LLAMA_VERSION} STREQUAL "LLAMA3")
    add_compile_definitions(USE_LLAMA3)
else()
    # Default to LLAMA2
    add_compile_definitions(USE_LLAMA2)
    message(WARNING "Invalid value for LLAMA_VERSION. Defaulting to LLAMA2")
endif()

add_library(llama2_cpp_dummy_lib STATIC ${LLAMA2_LIB_SOURCES})
target_include_directories(llama2_cpp_dummy_lib PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(llama2_cpp_dummy_lib glog::glog)

add_executable(llama2_cpp_dummy ${CMAKE_SOURCE_DIR}/main.cpp)
target_include_directories(llama2_cpp_dummy PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(llama2_cpp_dummy llama2_cpp_dummy_lib)

add_executable(llama2_reference ${CMAKE_SOURCE_DIR}/references/reference_llama2.cpp)
target_compile_definitions(llama2_reference PUBLIC -DINCLUDE_MAIN)



enable_testing()
add_subdirectory(tests)