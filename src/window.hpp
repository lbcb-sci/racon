/*!
 * @file window.hpp
 *
 * @brief Window class header file
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "spoa/alignment_engine.hpp"

namespace racon {

enum class WindowType {
  kNGS,  // Next Generation Sequencing
  kTGS  // Third Generation Sequencing
};

class Window {
 public:
  Window(
      std::uint64_t id,
      std::uint32_t rank,
      WindowType type,
      const char* backbone, std::uint32_t backbone_len,
      const char* quality, std::uint32_t quality_len);

  Window(const Window&) = default;
  Window& operator=(const Window&) = default;

  Window(Window&&) = default;
  Window& operator=(Window&&) = default;

  ~Window() = default;

  std::uint64_t id() const {
    return id_;
  }
  std::uint32_t rank() const {
    return rank_;
  }
  const std::string& consensus() const {
    return consensus_;
  }

  bool GenerateConsensus(
      std::shared_ptr<spoa::AlignmentEngine> alignment_engine,
      bool trim);

  void AddLayer(
      const char* sequence, uint32_t sequence_len,
      const char* quality, uint32_t quality_len,
      uint32_t begin,
      uint32_t end);

#ifdef CUDA_ENABLED
  friend class CUDABatchProcessor;
#endif

 private:
  std::uint64_t id_;
  std::uint32_t rank_;
  WindowType type_;
  std::string consensus_;
  std::vector<std::pair<const char*, std::uint32_t>> sequences_;
  std::vector<std::pair<const char*, std::uint32_t>> qualities_;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> positions_;
};

}  // namespace racon
