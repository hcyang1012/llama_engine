cmake_minimum_required(VERSION 3.0.0)
project(llama2_cpp_dummy VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(SOURCES 
    src/config.cpp
    src/transformer.cpp
    src/weights.cpp
    main.cpp
)

add_executable(llama2_cpp_dummy ${SOURCES})
target_include_directories(llama2_cpp_dummy PRIVATE include)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
