#pragma once

#include <cstddef>
#include <cstdint>
namespace llama2 {
using llama_size_t = std::size_t;

// signed integer types
using llama_int8_t = int8_t;
using llama_int32_t = int32_t;
using llama_int64_t = int64_t;

// unsigned integer types
using llama_uint8_t = uint8_t;
using llama_uint32_t = uint32_t;
using llama_uint64_t = uint64_t;

using llama_float = float;

}  // namespace llama2