/**
 * @file sampler.hpp
 * @brief  Header file for the Sampler class.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-07
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <cstdint>
#include <vector>

// Project Headers

// Third-party Headers

namespace llama2 {

class Sampler {
 public:
  /// @brief Struct used when sorting probabilities during top-p sampling
  struct ProbIdx {
    float prob;
    int idx;
  };

  Sampler(const size_t vocab_size, const float temperature, const float topp,
          const uint64_t rng_seed)
      : vocab_size(vocab_size),
        temperature(temperature),
        topp(topp),
        rng_state(rng_seed),
        prob_indices(vocab_size) {}

  const size_t &VocabSize() const { return vocab_size; }
  const auto &ProbIndices() const { return prob_indices; }
  const float &Temperature() const { return temperature; }
  const float &TopP() const { return topp; }
  const uint64_t &RngState() const { return rng_state; }

 private:
  size_t vocab_size;
  std::vector<ProbIdx> prob_indices;
  const float temperature;
  const float topp;
  const uint64_t rng_state;
};
}  // namespace llama2
