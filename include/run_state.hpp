/**
 * @file run_state.hpp
 * @brief RunState for the forward pass of the model.
 * @author Heecheol Yang (heecheol.yang@outlook.com)
 * @date 2024-05-03
 */

#pragma once

// C System-Headers

// C++ System-Headers
#include <memory>
// Project Headers
#include <tensor.hpp>

// Third-party Headers

namespace llama2 {

template <typename T> class RunState {
public:
  RunState() = default;

  void SetX(std::shared_ptr<Tensor<T>> tensor) { x = tensor; }
  void SetXB(std::shared_ptr<Tensor<T>> tensor) { xb = tensor; }
  void SetXB2(std::shared_ptr<Tensor<T>> tensor) { xb2 = tensor; }
  void SetHB(std::shared_ptr<Tensor<T>> tensor) { hb = tensor; }
  void SetHB2(std::shared_ptr<Tensor<T>> tensor) { hb2 = tensor; }
  void SetQ(std::shared_ptr<Tensor<T>> tensor) { q = tensor; }
  void SetK(std::shared_ptr<Tensor<T>> tensor) { k = tensor; }
  void SetV(std::shared_ptr<Tensor<T>> tensor) { v = tensor; }
  void SetAtt(std::shared_ptr<Tensor<T>> tensor) { att = tensor; }
  void SetLogits(std::shared_ptr<Tensor<T>> tensor) { logits = tensor; }
  void SetKeyCache(std::shared_ptr<Tensor<T>> tensor) { key_cache = tensor; }
  void SetValueCache(std::shared_ptr<Tensor<T>> tensor) {
    value_cache = tensor;
  }

  std::shared_ptr<Tensor<T>> X() { return x; }
  std::shared_ptr<Tensor<T>> XB() { return xb; }
  std::shared_ptr<Tensor<T>> XB2() { return xb2; }
  std::shared_ptr<Tensor<T>> HB() { return hb; }
  std::shared_ptr<Tensor<T>> HB2() { return hb2; }
  std::shared_ptr<Tensor<T>> Q() { return q; }
  std::shared_ptr<Tensor<T>> K() { return k; }
  std::shared_ptr<Tensor<T>> V() { return v; }
  std::shared_ptr<Tensor<T>> Att() { return att; }
  std::shared_ptr<Tensor<T>> Logits() { return logits; }
  std::shared_ptr<Tensor<T>> KeyCache() { return key_cache; }
  std::shared_ptr<Tensor<T>> ValueCache() { return value_cache; }

private:
  std::shared_ptr<Tensor<T>> x;  ///< activation at current time stamp (dim,)
  std::shared_ptr<Tensor<T>> xb; ///< same, but inside a residual branch (dim,)
  std::shared_ptr<Tensor<T>>
      xb2; ///< an additional buffer just for convenience (dim,)
  std::shared_ptr<Tensor<T>>
      hb; ///< buffer for hidden dimension in the ffn (hidden_dim,)
  std::shared_ptr<Tensor<T>>
      hb2; ///< buffer for hidden dimension in the ffn (hidden_dim,)
  std::shared_ptr<Tensor<T>> q; ///< query (dim,)
  std::shared_ptr<Tensor<T>> k; ///< key (dim,)
  std::shared_ptr<Tensor<T>> v; ///< value (dim,)
  std::shared_ptr<Tensor<T>>
      att; ///< buffer for scores/attention values (n_heads, seq_len)
  std::shared_ptr<Tensor<T>> logits; ///< output logits
  ///< kv cache
  std::shared_ptr<Tensor<T>> key_cache;   ///< (layer, seq_len, dim)
  std::shared_ptr<Tensor<T>> value_cache; ///< (layer, seq_len, dim)
};

} // namespace llama2
