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
#include <config.hpp>
#include <tensor.hpp>

// Third-party Headers

namespace llama2 {

template <typename T>
class RunState {
 public:
  RunState(const Config& config) : config_(config) {
    size_t kKVDims = (config.Dim() * config.NumKVHeads()) / config.NumHeads();
    x = std::make_shared<Tensor<T>>(Shape{static_cast<size_t>(config.Dim())});
    xb = std::make_shared<Tensor<T>>(Shape{static_cast<size_t>(config.Dim())});
    xb2 = std::make_shared<Tensor<T>>(Shape{static_cast<size_t>(config.Dim())});
    hb = std::make_shared<Tensor<T>>(
        Shape{static_cast<size_t>(config.HiddenDim())});
    hb2 = std::make_shared<Tensor<T>>(
        Shape{static_cast<size_t>(config.HiddenDim())});
    q = std::make_shared<Tensor<T>>(Shape{static_cast<size_t>(config.Dim())});
    key_cache = std::make_shared<Tensor<T>>(
        Shape{static_cast<size_t>(config.NumLayers()),
              static_cast<size_t>(config.SeqLen()), kKVDims});
    value_cache = std::make_shared<Tensor<T>>(
        Shape{static_cast<size_t>(config.NumLayers()),
              static_cast<size_t>(config.SeqLen()), kKVDims});
    att = std::make_shared<Tensor<T>>(
        Shape{static_cast<size_t>(config.NumHeads()),
              static_cast<size_t>(config.SeqLen())});
    logits = std::make_shared<Tensor<T>>(
        Shape{static_cast<size_t>(config.VocabSize())});

    if (!x || !xb || !xb2 || !hb || !hb2 || !q || !key_cache || !value_cache ||
        !att || !logits) {
      throw std::runtime_error("Failed to allocate memory for RunState.");
    }
  }

  std::shared_ptr<Tensor<T>> X() { return x; }
  std::shared_ptr<Tensor<T>> XB() { return xb; }
  std::shared_ptr<Tensor<T>> XB2() { return xb2; }
  std::shared_ptr<Tensor<T>> HB() { return hb; }
  std::shared_ptr<Tensor<T>> HB2() { return hb2; }
  std::shared_ptr<Tensor<T>> Q() { return q; }
  void UpdateKV(const size_t layer, const size_t pos) {
    const size_t kKVDims =
        (config_.Dim() * config_.NumKVHeads()) / config_.NumHeads();
    const size_t kLayerOffset = layer * config_.SeqLen() * kKVDims;
    k = key_cache->GetData() + kLayerOffset + pos * kKVDims;
    v = value_cache->GetData() + kLayerOffset + pos * kKVDims;
  }
  T* K() { return k; }
  T* V() { return v; }
  std::shared_ptr<Tensor<T>> Att() { return att; }
  std::shared_ptr<Tensor<T>> Logits() { return logits; }
  std::shared_ptr<Tensor<T>> KeyCache() { return key_cache; }
  std::shared_ptr<Tensor<T>> ValueCache() { return value_cache; }

 private:
  const Config& config_;
  std::shared_ptr<Tensor<T>> x;   ///< activation at current time stamp (dim,)
  std::shared_ptr<Tensor<T>> xb;  ///< same, but inside a residual branch (dim,)
  std::shared_ptr<Tensor<T>>
      xb2;  ///< an additional buffer just for convenience (dim,)
  std::shared_ptr<Tensor<T>>
      hb;  ///< buffer for hidden dimension in the ffn (hidden_dim,)
  std::shared_ptr<Tensor<T>>
      hb2;  ///< buffer for hidden dimension in the ffn (hidden_dim,)
  std::shared_ptr<Tensor<T>> q;  ///< query (dim,)
  T* k;                          ///< key (dim,), Pointer to the key cache
  T* v;                          ///< value (dim,), Pointer to the value cache
  std::shared_ptr<Tensor<T>>
      att;  ///< buffer for scores/attention values (n_heads, seq_len)
  std::shared_ptr<Tensor<T>> logits;  ///< output logits
  ///< kv cache
  std::shared_ptr<Tensor<T>> key_cache;    ///< (layer, seq_len, dim)
  std::shared_ptr<Tensor<T>> value_cache;  ///< (layer, seq_len, dim)
};

}  // namespace llama2
