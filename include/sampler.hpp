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
      : vocab_size(vocab_size), temperature(temperature), topp(topp),
        rng_state(rng_seed), prob_indices(vocab_size) {}

private:
  size_t vocab_size;
  std::vector<ProbIdx> prob_indices;
  float temperature;
  float topp;
  uint64_t rng_state;
};
} // namespace llama2
