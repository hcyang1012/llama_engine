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
#include <chrono>
#include <cstddef>
#include <fstream>
#include <memory>

// Project Headers
#include <config.hpp>
#include <decoder.hpp>
#include <encoder.hpp>
#include <op.hpp>
#include <run_state.hpp>
#include <sampler.hpp>
#include <tensor.hpp>
#include <tokenizer.hpp>
#include <weights.hpp>

// Third-party Headers

namespace llama {

struct RunConfig {
  float temperature = 1.0f;
  float topp = 0.9f;
  unsigned long long rng_seed = 0;
};
template <typename T>
class Transformer {
 public:
  Transformer(const TransformerConfig &config,
              const TransformerWeights<T> &weights, OpSet &op_set,
              const SpecialTokens &special_tokens)
      : config_(config),
        weights_(weights),
        op_set_(op_set),
        special_tokens_(special_tokens),
        run_state_(
            std::make_unique<RunState<T>>(config, op_set.GetDeviceType())) {}

  ~Transformer() {}

  const auto &GetConfig() const { return config_; }
  const auto &GetWeights() const { return weights_; }
  const auto &GetRunState() const { return run_state_; }
  auto &GetRunState() { return *run_state_; }

  std::string Generate(const Tokenizer<T> &tokenizer, const std::string &prompt,
                       const size_t steps, const RunConfig &run_config,
                       const bool print = true) {
    auto sampler = std::make_unique<Sampler>(
        config_.VocabSize(), run_config.temperature, run_config.topp,
        run_config.rng_seed, op_set_);
    std::stringstream ss;
    const auto encoder =
        Encoder<T>(tokenizer, prompt, true, false, special_tokens_);
    const auto &prompt_tokens = encoder.PromptTokens();

    auto token = prompt_tokens[0];
    size_t pos = 0;
    int next;
    bool is_first = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (pos < steps) {
      auto logits = Forward(token, pos);
      if (pos < prompt_tokens.size() - 1) {
        // Prefill stage
        next = prompt_tokens[pos + 1];
      } else {
        // Generation stage
        next = sampler->Sample(logits);
      }
      pos++;

      if (next == special_tokens_.GetToken(SpecialTokens::Idx::IDX_BOS_01)) {
        break;
      }

      auto piece = Decoder<T>::Decode(tokenizer, token, next, special_tokens_);
      safe_print(piece, ss, print);
      token = next;
      // ignore the first token for time calculation to ignore the warm-up
      // time
      if (is_first) {
        start_time = std::chrono::high_resolution_clock::now();
        is_first = false;
      }
    }
    if (pos > 1) {
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          end_time - start_time);
      if (print == true) {
        std::cout << std::endl;
        std::cout << "Average tok/s: "
                  << (pos - 1) / (double)(duration.count() / 1000) << std::endl;
      }
    }
    return ss.str();
  }

  const Tensor<T> Forward(const size_t token, const size_t pos) {
    const size_t kInputEmbedDim = config_.Dim();
    const size_t kPerHeadDim = kInputEmbedDim / config_.NumHeads();
    const size_t kKVDim =
        (kInputEmbedDim * config_.NumKVHeads()) / config_.NumHeads();
    const size_t kKVMul = config_.NumHeads() / config_.NumKVHeads();

    auto embed = static_cast<T *>(weights_.TokenEmbeddingTable()->GetBuffer()) +
                 token * kInputEmbedDim;
    DeviceFactory::GetDevice(op_set_.GetDeviceType())
        .GetMemcpy()
        .Copy(*(run_state_->X().GetData()), (void *)embed,
              kInputEmbedDim * sizeof(T));

    auto Q = run_state_->Q();
    auto X = run_state_->X();
    auto XB = run_state_->XB();
    auto XB2 = run_state_->XB2();
    auto HB = run_state_->HB();
    auto HB2 = run_state_->HB2();

    for (size_t layer = 0; layer < config_.NumLayers(); ++layer) {
      // RMSNorm
      {
        const auto kRmsAttWeight = weights_.RMSAttnWeight(layer);
        op_set_.RmsNorm<T>(X, kRmsAttWeight, XB);
      }

      auto K = run_state_->K(layer, pos).ReShape({kKVDim});
      auto V = run_state_->V(layer, pos).ReShape({kKVDim});

      // Calculate Q, K, V
      {
        op_set_.MatMul<T>(
            weights_.WQ(layer).ReShape({kInputEmbedDim, kInputEmbedDim}),
            run_state_->XB(), Q);

        op_set_.MatMul<T>(weights_.WK(layer).ReShape({kInputEmbedDim, kKVDim}),
                          run_state_->XB(), K);

        op_set_.MatMul<T>(weights_.WV(layer).ReShape({kInputEmbedDim, kKVDim}),
                          run_state_->XB(), V);
      }

      // RoPE
      { op_set_.RoPE<T>(pos, config_, Q, K); }

#if 0
      // Multi-Head Attention
      const size_t kNumHeads = config_.NumHeads();
      const size_t kKVMul = config_.KVMul();
      for (size_t head_idx = 0; head_idx < kNumHeads; ++head_idx) {
        const size_t kKVHeadIdx = head_idx / kKVMul;
        auto Q = run_state_->Q(head_idx);
        auto K_layer = run_state_->K(layer);
        auto V_layer = run_state_->V(layer);

        auto XB = run_state_->XB(head_idx);
        op_set_.Attention<T>(Q, K_layer, V_layer, config_, pos, kKVHeadIdx, XB);
      }
#else
      op_set_.MultiAttention<T>(layer, pos, config_, *run_state_);
#endif
      // Matmul
      {
        const auto WO =
            weights_.WO(layer).ReShape({kInputEmbedDim, kInputEmbedDim});
        op_set_.MatMul<T>(WO, run_state_->XB(), XB2);
      }

      // Residual Connection
      { op_set_.ElementwiseAdd<T>(X, XB2, X); }

      // Feed Forward Network RMSNorm
      {
        const auto WRMSFFN = weights_.RMSFFNWeight(layer);
        op_set_.RmsNorm(X, WRMSFFN, XB);
      }

      // SWiGLU Feed Forward Network
      {
        const auto W1 = weights_.W1(layer);
        const auto W2 = weights_.W2(layer);
        const auto W3 = weights_.W3(layer);

        op_set_.MatMul<T>(W1, XB, HB);
        op_set_.MatMul<T>(W3, XB, HB2);

        op_set_.SiLU_EWMul<T>(HB, HB2, HB);

        op_set_.MatMul<T>(W2, HB, XB);
      }

      // Residual Connection
      { op_set_.ElementwiseAdd<T>(X, XB, X); }
    }

    // Final RMSNorm
    const auto kRmsFinalWeight = weights_.RMSFinalWeight();
    { op_set_.RmsNorm<T>(X, kRmsFinalWeight, X); }

    // Logits
    { op_set_.MatMul<T>(weights_.WCLS(), X, run_state_->Logits()); }

    const auto logits = run_state_->Logits();
    volatile const T *logits_ptr =
        static_cast<volatile T *>(logits.GetData()->GetBuffer());
    return logits;
  }

 private:
  void safe_print(const std::string &str, std::ostream &os, const bool print) {
    if (str.size() > 0) {
      unsigned char byte = str[0];
      if (std::isprint(byte) || std::isspace(byte)) {
        os << str;
        if (print) {
          std::cout << str << std::flush;
        }
      }
    }
  }

  const TransformerConfig &config_;  ///< Hyperparameters of the Transformer
  const TransformerWeights<T> &weights_;  ///< Weights of the Transformer
  OpSet &op_set_;
  const SpecialTokens &special_tokens_;

  std::unique_ptr<RunState<T>> run_state_;
};

}  // namespace llama
