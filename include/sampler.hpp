/**
 * @file sampler.hpp
 * @brief  Header file for the Sampler class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-07
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

// Project Headers
#include <op.hpp>
#include <rng.hpp>
#include <tensor.hpp>

// Third-party Headers

namespace llama {

class Sampler {
 public:
  /// @brief Struct used when sorting probabilities during top-p sampling
  struct ProbIdx {
    float prob;
    int idx;

    // Sort in descending order
    bool operator<(const ProbIdx &rhs) const { return prob > rhs.prob; }
  };

  Sampler(const size_t vocab_size, const float temperature, const float topp,
          const uint64_t rng_seed, OpSet &op_set)
      : vocab_size_(vocab_size),
        temperature_(temperature),
        topp_(topp),
        prob_indices_(vocab_size),
        rng_(std::make_unique<ReferenceRng>(rng_seed)),
        op_set_(op_set) {}

  const size_t &VocabSize() const { return vocab_size_; }
  const auto &ProbIndices() const { return prob_indices_; }
  const float &Temperature() const { return temperature_; }
  const float &TopP() const { return topp_; }

  template <typename T>
  int Sample(const Tensor<T> &logits) {
    DCHECK_EQ(logits.GetShape().GetRank(), 1)
        << "Input tensor should be 1D tensor";
    DCHECK_EQ(logits.GetShape()[0], VocabSize())
        << "Input tensor should have the same size as the vocab size";
    int next;
    if (Temperature() == 0.0f) {
      next = op_set_.ArgMax<T>(logits);
    } else {
      Tensor<T> logits_copy(logits.GetShape());
      for (size_t q = 0; q < VocabSize(); q++) {
        logits_copy[{q}] = logits[{q}] / Temperature();
      }
      op_set_.SoftMax<T>(logits_copy, logits_copy);
      float coin = rng_->RandomF32();

      if (TopP() <= 0 || TopP() >= 1) {
        next = sample_multiple(logits_copy, coin);
      } else {
        next = sample_top_p(logits_copy, coin);
      }
    }
    return next;
  }

 private:
  size_t vocab_size_;
  std::vector<ProbIdx> prob_indices_;
  const float temperature_;
  const float topp_;

  std::unique_ptr<Rng> rng_;

  OpSet &op_set_;

  template <typename T>
  int sample_multiple(const Tensor<T> &probabilities, const float coin) {
    DCHECK_EQ(probabilities.GetShape().GetRank(), 1)
        << "Input tensor should be 1D tensor";
    float cdf = 0.0f;
    for (size_t i = 0; i < probabilities.GetShape()[0]; i++) {
      cdf += probabilities[{i}];
      if (coin < cdf) {
        return i;
      }
    }
    return probabilities.GetShape()[0] - 1;
  }

  template <typename T>
  int sample_top_p(const Tensor<T> &probs, const float coin) {
    DCHECK_EQ(probs.GetShape().GetRank(), 1)
        << "Input tensor should be 1D tensor";
    const size_t N = VocabSize();

    int n0 = 0;
    const float cutoff = (1.0f - TopP()) / (N - 1);

    for (size_t i = 0; i < N; i++) {
      if (probs[{i}] >= cutoff) {
        prob_indices_[n0].idx = i;
        prob_indices_[n0].prob = probs[{i}];
        n0++;
      }
    }

    std::sort(prob_indices_.begin(), prob_indices_.begin() + n0);

    float cdf = 0.0f;
    size_t last_idx = n0 - 1;
    for (size_t i = 0; i < n0; i++) {
      cdf += prob_indices_[i].prob;
      if (cdf > TopP()) {
        last_idx = i;
        break;
      }
    }

    float r = coin * cdf;
    cdf = 0.0f;
    for (size_t i = 0; i <= last_idx; i++) {
      cdf += prob_indices_[i].prob;
      if (r < cdf) {
        return prob_indices_[i].idx;
      }
    }

    return prob_indices_[last_idx].idx;
  }
};
}  // namespace llama
