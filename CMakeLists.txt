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

# Download json lib from https://raw.githubusercontent.com/nlohmann/json/refs/heads/develop/single_include/nlohmann/json.hpp to 3rdparty/json/json.hpp
file(DOWNLOAD 
    https://raw.githubusercontent.com/nlohmann/json/refs/heads/develop/single_include/nlohmann/json.hpp 
    ${CMAKE_CURRENT_BINARY_DIR}/download_deps/json/json.hpp
)


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


set(DEFAULT_LINK_LIBS glog::glog)

# Add option to enable/disable cuda
option(USE_CUDA "Enable CUDA" OFF)
if(USE_CUDA)
    include(${CMAKE_SOURCE_DIR}/cmake/cuda_build.cmake)
    build_cuda_file(op_cuda ${CMAKE_SOURCE_DIR}/include/op_cuda.cu)

    function(append_cuda_deps target_name)
        target_compile_definitions(${target_name} PRIVATE -DUSE_CUDA)
        target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})
        # Update LINK_LIBS globally
        set(LINK_LIBS ${DEFAULT_LINK_LIBS} ${CUDA_LIBRARIES} op_cuda PARENT_SCOPE)
    endfunction()    
endif()

add_executable(llama2_cpp ${CMAKE_SOURCE_DIR}/llama2.cpp)
if(USE_CUDA)
    append_cuda_deps(llama2_cpp)
else()
    set(LINK_LIBS ${DEFAULT_LINK_LIBS})
endif()
target_include_directories(llama2_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/download_deps/)
target_link_libraries(llama2_cpp ${LINK_LIBS})



add_executable(llama3_cpp ${CMAKE_SOURCE_DIR}/llama3.cpp)
if(USE_CUDA)
    append_cuda_deps(llama3_cpp)
else()
    set(LINK_LIBS ${DEFAULT_LINK_LIBS})
endif()
target_include_directories(llama3_cpp PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/download_deps/)
target_link_libraries(llama3_cpp ${LINK_LIBS})


add_executable(llama2_reference ${CMAKE_SOURCE_DIR}/references/reference_llama2.cpp)
target_include_directories(llama2_reference PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(llama2_reference PUBLIC -DINCLUDE_MAIN)


add_executable(llama3_reference ${CMAKE_SOURCE_DIR}/references/reference_llama3.cpp)
target_include_directories(llama3_reference PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(llama3_reference PUBLIC -DINCLUDE_MAIN)


enable_testing()
add_subdirectory(tests)