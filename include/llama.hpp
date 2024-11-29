/**
 * @file lllama.hpp
 * @brief A wrapper for the llama to support multiple versions of the model.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-11-20
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <string>

// Project Headers
#include <dtypes.h>

#include <tokenizer.hpp>
#include <transformer.hpp>
// Third-party Headers

namespace llama {
template <typename T>
class LlamaModel {
 public:
  struct LlamaConfig {
    std::string checkpoint_path;
    std::string tokenizer_path;
    DeviceType device_type;
    float temperature;
    float topp;
    llama_uint32_t steps;
    unsigned long long rng_seed;
  };
  LlamaModel(const LlamaConfig& config)
      : config_(config),
        run_config_({config.temperature, config.topp, config.rng_seed}) {}
  ~LlamaModel() = default;

  std::string Generate(const std::string& prompt, const size_t steps) {
    if (!transformer_ || !tokenizer_) {
      model_init();
    }
    return transformer_->Generate(*tokenizer_, prompt, steps);
  }

  const auto& GetTransformer() const { return *transformer_; }

 protected:
  virtual void model_init() = 0;
  LlamaConfig config_;
  Transformer<float>::RunConfig run_config_;
  std::unique_ptr<Transformer<T>> transformer_;
  std::unique_ptr<Tokenizer<T>> tokenizer_;
  std::unique_ptr<SpecialTokens> special_tokens_;
  std::unique_ptr<OpSet> op_set_;
};

template <typename T>
class Llama2 : public LlamaModel<T> {
 public:
  Llama2(const typename LlamaModel<T>::LlamaConfig& config)
      : LlamaModel<T>(config) {}
  ~Llama2() = default;

 private:
  void model_init() override {
    this->op_set_ = CreateOpSet(this->config_.device_type);
    this->special_tokens_ = std::make_unique<SpecialTokensLlama2>();
    this->transformer_ = std::make_unique<Transformer<T>>(
        this->config_.checkpoint_path, this->run_config_, *(this->op_set_),
        *(this->special_tokens_));
    this->tokenizer_ = std::make_unique<Tokenizer<T>>(
        this->config_.tokenizer_path,
        this->transformer_->GetConfig().VocabSize());
  }
};
}  // namespace llama
