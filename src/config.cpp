/**
 * @file template.cpp
 * @brief This file is a template for creating new source files.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */
// C System-Headers

// C++ System-Headers

// Project Headers
#include <dtypes.h>

#include <config.hpp>
// Third-party Headers

namespace llama2 {

Config::Config(std::ifstream &config_file) { load_config(config_file); }

llama_uint32_t Config::Dim() const { return kDim; }

llama_uint32_t Config::HiddenDim() const { return kHiddenDim; }

llama_uint32_t Config::NumLayers() const { return kNumLayers; }

llama_uint32_t Config::NumHeads() const { return kNumHeads; }

llama_uint32_t Config::NumKVHeads() const { return kNumKVHeads; }

llama_uint32_t Config::HeadDim() const { return kHeadDim; }

llama_uint32_t Config::KVHeadDim() const { return kKVHeadDim; }

llama_uint32_t Config::KVMul() const { return kKVMul; }

llama_uint32_t Config::VocabSize() const { return kVocabSize; }

llama_uint32_t Config::SeqLen() const { return kSeqLen; }

void Config::load_config(std::ifstream &config_file) {
  config_file.read(reinterpret_cast<char *>(&kDim), sizeof(kDim));
  config_file.read(reinterpret_cast<char *>(&kHiddenDim), sizeof(kHiddenDim));
  config_file.read(reinterpret_cast<char *>(&kNumLayers), sizeof(kNumLayers));
  config_file.read(reinterpret_cast<char *>(&kNumHeads), sizeof(kNumHeads));
  config_file.read(reinterpret_cast<char *>(&kNumKVHeads), sizeof(kNumKVHeads));
  config_file.read(reinterpret_cast<char *>(&kVocabSize), sizeof(kVocabSize));
  config_file.read(reinterpret_cast<char *>(&kSeqLen), sizeof(kSeqLen));

  kVocabSize = VocabSize();
  kKVHeadDim = Dim() * NumKVHeads() / NumHeads();
  kHeadDim = Dim() / NumHeads();
  kKVMul = NumHeads() / NumKVHeads();
}

size_t Config::Size() {
  return sizeof(kDim) + sizeof(kHiddenDim) + sizeof(kNumLayers) +
         sizeof(kNumHeads) + sizeof(kNumKVHeads) + sizeof(kVocabSize) +
         sizeof(kSeqLen);
}

}  // namespace llama2
