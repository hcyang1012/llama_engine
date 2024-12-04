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
#include <dtypes.hpp>
#include <tokenizer.hpp>
#include <transformer.hpp>
// Third-party Headers

namespace llama {

struct LlamaConfig {
  std::string checkpoint_path;
  std::string tokenizer_path;
  DeviceType device_type;
};

template <typename T>
class LlamaModel {
 public:
  LlamaModel(const LlamaConfig& llama_config,
             std::unique_ptr<TransformerConfig> transformer_config,
             std::unique_ptr<OpSet> op_set,
             std::unique_ptr<SpecialTokens> special_tokens)
      : transformer_config_(std::move(transformer_config)),
        weights_(std::make_unique<TransformerWeights<T>>(
            llama_config.checkpoint_path, *transformer_config_,
            llama_config.device_type)),
        op_set_(std::move(op_set)),
        special_tokens_(std::move(special_tokens)),
        transformer_(std::make_unique<Transformer<T>>(
            *transformer_config_, *weights_, *op_set_, *special_tokens_)),
        tokenizer_(std::make_unique<Tokenizer<T>>(
            llama_config.tokenizer_path, transformer_config_->VocabSize())) {}

  ~LlamaModel() = default;

  std::string Generate(const std::string& prompt, const size_t steps,
                       const RunConfig& run_config) {
    if (!transformer_ || !tokenizer_) {
      throw std::runtime_error("Model is not initialized.");
    }
    return transformer_->Generate(*tokenizer_, prompt, steps, run_config);
  }

  void Chat(const std::string& prompt, const std::string& system_prompt,
            const size_t steps, const RunConfig& run_config) {
    if (!transformer_ || !tokenizer_) {
      throw std::runtime_error("Model is not initialized.");
    }
    transformer_->Chat(*tokenizer_, prompt, system_prompt, run_config, steps);
  }

  auto& GetTransformer() const { return *transformer_; }
  const auto& GetTokenizer() const { return *tokenizer_; }

 protected:
  std::unique_ptr<TransformerConfig> transformer_config_;
  std::unique_ptr<TransformerWeights<T>> weights_;
  std::unique_ptr<OpSet> op_set_;
  std::unique_ptr<SpecialTokens> special_tokens_;
  std::unique_ptr<Transformer<T>> transformer_;
  std::unique_ptr<Tokenizer<T>> tokenizer_;

 private:
  std::unique_ptr<TransformerConfig> load_config(const std::string& ckpt_file) {
    std::ifstream if_chkpt_file(ckpt_file, std::ios::binary);
    if (!if_chkpt_file.is_open()) {
      throw std::runtime_error("Failed to open the checkpoint file.");
    }

    return std::make_unique<TransformerConfig>(if_chkpt_file);
  }
};

template <typename T>
class Llama2 : public LlamaModel<T> {
 public:
  Llama2(const LlamaConfig& llama_config)
      : LlamaModel<T>(llama_config,
                      std::move(std::make_unique<ConfigLlama2>(
                          llama_config.checkpoint_path)),
                      std::move(CreateOpSet(llama_config.device_type)),
                      std::move(std::make_unique<SpecialTokensLlama2>())) {}
  ~Llama2() = default;

 protected:
};

template <typename T>
class Llama3 : public LlamaModel<T> {
 public:
  Llama3(const LlamaConfig& llama_config)
      : LlamaModel<T>(llama_config,
                      std::move(std::make_unique<ConfigLlama3>(
                          llama_config.checkpoint_path)),
                      std::move(CreateOpSet(llama_config.device_type)),
                      std::move(std::make_unique<SpecialTokensLlama3>())) {}
  ~Llama3() = default;

 protected:
};
}  // namespace llama
