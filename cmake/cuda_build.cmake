find_package(CUDA REQUIRED)

enable_language(CUDA)

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
endfunction()