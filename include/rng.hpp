#pragma once
#include <cstddef>

class Rng {
 public:
  Rng(const unsigned long long seed) : seed_(seed) {}
  // Random float in [0, 1)
  virtual float RandomF32() = 0;

 protected:
  const unsigned long long seed_;

 private:
};

class ReferenceRng : public Rng {
 public:
  ReferenceRng(const unsigned long long seed) : Rng(seed), state_(seed) {}

  float RandomF32() override {
    return static_cast<float>(RandomU32() >> 8) / 16777216.0f;
  }

 private:
  unsigned long long state_;

  unsigned int RandomU32() {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return (state_ * 0x2545F4914F6CDD1Dull) >> 32;
  }
};