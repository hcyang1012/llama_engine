/**
 * @file template.cpp
 * @brief This file is a template for creating new source files.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */
// C System-Headers

// C++ System-Headers

// Project Headers
#include <config.hpp>

// Third-party Headers

namespace llama2 {

Config::Config(const std::string &config_file) { load_config(config_file); }

size_t Config::Dim() const { return kDim; }

size_t Config::HiddenDim() const { return kHiddenDim; }

size_t Config::NumLayers() const { return kNumLayers; }

size_t Config::NumHeads() const { return kNumHeads; }

size_t Config::NumKVHeads() const { return kNumKVHeads; }

size_t Config::VocabSize() const { return kVocabSize; }

size_t Config::SeqLen() const { return kSeqLen; }

void Config::load_config(const std::string &config_file) {}

} // namespace llama2
