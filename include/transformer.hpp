/**
 * @file transformer.hpp
 * @brief Header for the Transformer class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// C++ System-Headers
#include <cstddef>
#include <fstream>
#include <memory>

// Project Headers
#include <config.hpp>
#include <encoder.hpp>
#include <op.hpp>
#include <run_state.hpp>
#include <sampler.hpp>
#include <tensor.hpp>
#include <tokenizer.hpp>
#include <weights.hpp>
// Third-party Headers

namespace llama2 {

template <typename T>
class Transformer {
 public:
  Transformer(const std::string &ckpt_file) { load_checkpoint(ckpt_file); }
  ~Transformer() {}

  const auto &GetConfig() const { return *config_; }
  const auto &GetWeights() const { return *weights_; }
  const auto &GetRunState() const { return *run_state_; }

  auto &GetRunState() { return *run_state_; }

  void Generate(const Tokenizer<T> &tokenizer, const Sampler &sampler,
                const std::string &prompt, const size_t steps) {
    const auto encoder = Encoder<T>(tokenizer, prompt, true, true);
    const auto &prompt_tokens = encoder.PromptTokens();

    auto token = prompt_tokens[0];
    size_t pos = 0;
    while (pos < steps) {
      // forward(token, pos);
      pos++;
    }
  }

  const Tensor<T> Forward(const size_t token, const size_t pos) {
    const size_t kInputEmbedDim = config_->Dim();
    const size_t kPerHeadDim = kInputEmbedDim / config_->NumHeads();
    const size_t kKVDim =
        (kInputEmbedDim * config_->NumKVHeads()) / config_->NumHeads();
    const size_t kKVMul = config_->NumHeads() / config_->NumKVHeads();

    auto embed = weights_->TokenEmbeddingTable() + token * kInputEmbedDim;
    std::copy(embed, embed + kInputEmbedDim, run_state_->X().GetData());

    for (size_t layer = 0; layer < config_->NumLayers(); ++layer) {
      // RMSNorm
      {
        const auto kRmsAttWeight = weights_->RMSAttnWeight(layer);
        RmsNorm<T>::Compute(run_state_->X(), kRmsAttWeight, run_state_->XB());
      }

      // Calculate Q, K, V
      {
        auto &Q = run_state_->Q();
        auto K = run_state_->K(layer, pos).ReShape({kKVDim});
        auto V = run_state_->V(layer, pos).ReShape({kKVDim});
        MatMul<T>::Compute(
            weights_->WQ(layer).ReShape({kInputEmbedDim, kInputEmbedDim}),
            run_state_->XB(), Q);
        MatMul<T>::Compute(
            weights_->WK(layer).ReShape({kInputEmbedDim, kKVDim}),
            run_state_->XB(), K);
        MatMul<T>::Compute(
            weights_->WV(layer).ReShape({kInputEmbedDim, kKVDim}),
            run_state_->XB(), V);
      }

      // RoPE
      {
        auto &Q = run_state_->Q();
        RoPE<T>::Compute(Q, pos, *config_, Q, true);
      }

      // Multi-Head Attention
      for (size_t head_idx = 0; head_idx < config_->NumHeads(); ++head_idx) {
        const size_t kKVHeadIdx = head_idx / config_->KVMul();
        auto Q = run_state_->Q(head_idx);
        auto K = run_state_->K(layer);
        auto V = run_state_->V(layer);

        auto XB = run_state_->XB(head_idx);
        Attention<T>::Compute(Q, K, V, *config_, pos, kKVHeadIdx, XB);
      }

      // Matmul
      {
        const auto XB = run_state_->XB();
        const auto WO =
            weights_->WO(layer).ReShape({kInputEmbedDim, kInputEmbedDim});
        auto XB2 = run_state_->XB2();
        MatMul<T>::Compute(WO, XB, XB2);
      }

      // Residual Connection
      {
        auto X = run_state_->X();
        const auto &XB2 = run_state_->XB2();
        ElementwiseAdd<T>::Compute(X, XB2, X);
      }

      // Feed Forward Network RMSNorm
      {
        RmsNorm<T>::Compute(run_state_->X(), weights_->RMSFFNWeight(layer),
                            run_state_->XB());
      }

      // SWiGLU Feed Forward Network
      {
        MatMul<T>::Compute(weights_->W1(layer), run_state_->XB(),
                           run_state_->HB());
        MatMul<T>::Compute(weights_->W3(layer), run_state_->XB(),
                           run_state_->HB2());

        SiLU_EWMul<T>::Compute(run_state_->HB(), run_state_->HB2(),
                               run_state_->HB());

        MatMul<T>::Compute(weights_->W2(layer), run_state_->HB(),
                           run_state_->XB());
      }

      // Residual Connection
      {
        auto &X = run_state_->X();
        const auto &XB = run_state_->XB();
        ElementwiseAdd<T>::Compute(X, XB, X);
      }
    }

    // Final RMSNorm
    {
      const auto kRmsFinalWeight = weights_->RMSFinalWeight();
      RmsNorm<T>::Compute(run_state_->X(), kRmsFinalWeight, run_state_->X());
    }

    // Logits
    {
      MatMul<T>::Compute(weights_->WCLS(), run_state_->X(),
                         run_state_->Logits());
    }

    return run_state_->Logits();
  }

 private:
  void load_checkpoint(const std::string &ckpt_file) {
    // Load the configuration file
    std::ifstream if_chkpt_file(ckpt_file, std::ios::binary);
    if (!if_chkpt_file.is_open()) {
      throw std::runtime_error("Failed to open the checkpoint file.");
    }

    // Load the configuration
    config_ = std::make_unique<Config>(if_chkpt_file);

    // Load the weights
    if_chkpt_file.seekg(0, std::ios::end);
    file_size_ = if_chkpt_file.tellg();
    if_chkpt_file.close();

    fd_ = open(ckpt_file.c_str(), O_RDONLY);
    if (fd_ == -1) {
      throw std::runtime_error("Failed to open the checkpoint file.");
    }

    mapped_file_ = static_cast<T *>(
        mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));

    if (mapped_file_ == MAP_FAILED) {
      throw std::runtime_error("Failed to map the checkpoint file.");
    }

    T *weights_ptr = mapped_file_ + Config::Size() / sizeof(T);

    load_weights(weights_ptr);
    run_state_ = std::make_unique<RunState<T>>(*config_);
  }

  void load_weights(T *weights_ptr) {
    weights_ = std::make_unique<TransformerWeights<T>>(*config_, weights_ptr);
  }

  std::unique_ptr<Config> config_;  ///< Hyperparameters of the Transformer
  std::unique_ptr<TransformerWeights<T>>
      weights_;                             ///< Weights of the Transformer
  std::unique_ptr<RunState<T>> run_state_;  ///< Run state of the Transformer

  int fd_;             // file descriptor for the memory mapped file
  ssize_t file_size_;  // size of the memory mapped file
  T *mapped_file_;     // pointer to the memory mapped file
};

}  // namespace llama2
