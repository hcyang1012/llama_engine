find_package(CUDA REQUIRED)

enable_language(CUDA)

set(TE_PATH "/workspace/TransformerEngine")
set(TE_LIB_PATH "${TE_PATH}/build/lib.linux-x86_64-3.10/transformer_engine")
set(TE_INCLUDE_PATH "${TE_PATH}/transformer_engine/common/include")

find_library(TE_LIB NAMES transformer_engine PATHS "${TE_LIB_PATH}/transformer_engine" ${TE_LIB_PATH} ENV TE_LIB_PATH REQUIRED)
message(STATUS "Found transformer_engine: ${TE_LIB}")

function (append_cuda_deps target_name)
    target_compile_definitions(${target_name} PRIVATE -DUSE_CUDA)
    target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES} )
endfunction()

function (build_cuda_file target_name)
    add_library(${target_name} ${ARGN})
    target_compile_definitions(${target_name} PRIVATE -DUSE_CUDA)
    target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES} )
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES 52 60 70 75 80)    

    target_link_libraries(${target_name} PRIVATE ${TE_LIB}  nvrtc cudnn)
    target_include_directories(${target_name} PRIVATE ${TE_INCLUDE_PATH})
endfunction()