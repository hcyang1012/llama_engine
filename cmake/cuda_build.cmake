find_package(CUDA REQUIRED)



function (append_cuda_deps target_name)
    enable_language(CUDA)
    target_compile_definitions(${target_name} PRIVATE -DUSE_CUDA)
    target_include_directories(${target_name} PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES} )
endfunction()