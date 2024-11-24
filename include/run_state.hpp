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

  Tensor<T>& X() { return *x; }
  Tensor<T>& XB() { return *xb; }
  Tensor<T> XB(const size_t head_idx) {
    CHECK_LT(head_idx, config_.NumHeads())
        << "Head index should be less than the number of heads";

    return Tensor<T>(xb->GetData() + head_idx * config_.HeadDim(),
                     {static_cast<size_t>(config_.HeadDim())});
  }
  Tensor<T>& XB2() { return *xb2; }
  Tensor<T>& HB() { return *hb; }
  Tensor<T>& HB2() { return *hb2; }
  Tensor<T>& Q() { return *q; }
  Tensor<T> Q(const size_t head_idx) {
    CHECK_LT(head_idx, config_.NumHeads())
        << "Head index should be less than the number of heads";

    return Tensor<T>(Q().GetData() + head_idx * config_.HeadDim(),
                     {static_cast<size_t>(config_.HeadDim())});
  }

  Tensor<T> K() {
    return Tensor<T>(key_cache->GetData(),
                     {static_cast<size_t>(config_.HeadDim()),
                      static_cast<size_t>(config_.NumKVHeads()),
                      static_cast<size_t>(config_.SeqLen()),
                      static_cast<size_t>(config_.NumLayers())});
  }
  Tensor<T> K(const size_t layer) {
    return Tensor<T>(key_cache->GetData() +
                         (layer * (static_cast<size_t>(config_.HeadDim()) *
                                   static_cast<size_t>(config_.NumKVHeads()) *
                                   static_cast<size_t>(config_.SeqLen()))),
                     {static_cast<size_t>(config_.HeadDim()),
                      static_cast<size_t>(config_.NumKVHeads()),
                      static_cast<size_t>(config_.SeqLen())});
  }
  Tensor<T> K(const size_t layer, const size_t pos) {
    return Tensor<T>(key_cache->GetData() +
                         (layer * (static_cast<size_t>(config_.HeadDim()) *
                                   static_cast<size_t>(config_.NumKVHeads()) *
                                   static_cast<size_t>(config_.SeqLen()))) +
                         (pos * (static_cast<size_t>(config_.HeadDim()) *
                                 static_cast<size_t>(config_.NumKVHeads()))),
                     {static_cast<size_t>(config_.HeadDim()),
                      static_cast<size_t>(config_.NumKVHeads())});
  }

  Tensor<T> K(const size_t layer, const size_t pos, const size_t head_idx) {
    return Tensor<T>(key_cache->GetData() +
                         (layer * (static_cast<size_t>(config_.HeadDim()) *
                                   static_cast<size_t>(config_.NumKVHeads()) *
                                   static_cast<size_t>(config_.SeqLen()))) +
                         (pos * (static_cast<size_t>(config_.HeadDim()) *
                                 static_cast<size_t>(config_.NumKVHeads()))) +
                         (head_idx * (static_cast<size_t>(config_.HeadDim()))),
                     {
                         static_cast<size_t>(config_.HeadDim()),
                     });
  }

  Tensor<T> V() {
    return Tensor<T>(value_cache->GetData(),
                     {static_cast<size_t>(config_.HeadDim()),
                      static_cast<size_t>(config_.NumKVHeads()),
                      static_cast<size_t>(config_.SeqLen()),
                      static_cast<size_t>(config_.NumLayers())});
  }
  Tensor<T> V(const size_t layer) {
    return Tensor<T>(value_cache->GetData() +
                         (layer * (static_cast<size_t>(config_.HeadDim()) *
                                   static_cast<size_t>(config_.NumKVHeads()) *
                                   static_cast<size_t>(config_.SeqLen()))),
                     {static_cast<size_t>(config_.HeadDim()),
                      static_cast<size_t>(config_.NumKVHeads()),
                      static_cast<size_t>(config_.SeqLen())});
  }
  Tensor<T> V(const size_t layer, const size_t pos) {
    return Tensor<T>(value_cache->GetData() +
                         (layer * (static_cast<size_t>(config_.HeadDim()) *
                                   static_cast<size_t>(config_.NumKVHeads()) *
                                   static_cast<size_t>(config_.SeqLen()))) +
                         (pos * (static_cast<size_t>(config_.HeadDim()) *
                                 static_cast<size_t>(config_.NumKVHeads()))),
                     {static_cast<size_t>(config_.HeadDim()),
                      static_cast<size_t>(config_.NumKVHeads())});
  }

  Tensor<T> V(const size_t layer, const size_t pos, const size_t head_idx) {
    return Tensor<T>(value_cache->GetData() +
                         (layer * (static_cast<size_t>(config_.HeadDim()) *
                                   static_cast<size_t>(config_.NumKVHeads()) *
                                   static_cast<size_t>(config_.SeqLen()))) +
                         (pos * (static_cast<size_t>(config_.HeadDim()) *
                                 static_cast<size_t>(config_.NumKVHeads()))) +
                         (head_idx * (static_cast<size_t>(config_.HeadDim()))),
                     {
                         static_cast<size_t>(config_.HeadDim()),
                     });
  }

  Tensor<T>& Att() { return *att; }
  Tensor<T> Att(const size_t head_idx) {
    return Tensor<T>(att->GetData() + head_idx * config_.SeqLen(),
                     {static_cast<size_t>(config_.SeqLen())});
  }
  Tensor<T>& Logits() { return *logits; }
  Tensor<T>& KeyCache() { return *key_cache; }
  Tensor<T>& ValueCache() { return *value_cache; }

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
