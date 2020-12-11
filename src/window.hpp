// Copyright (c) 2020 Robert Vaser

#ifndef RACON_WINDOW_HPP_
#define RACON_WINDOW_HPP_

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

struct Window {
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

  void GenerateConsensus(
      std::shared_ptr<spoa::AlignmentEngine> alignment_engine,
      bool trim);

  void AddLayer(
      const char* sequence, uint32_t sequence_len,
      const char* quality, uint32_t quality_len,
      uint32_t begin,
      uint32_t end);

  std::uint64_t id;
  std::uint32_t rank;
  WindowType type;
  bool status;
  std::string consensus;
  std::vector<std::pair<const char*, std::uint32_t>> sequences;
  std::vector<std::pair<const char*, std::uint32_t>> qualities;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> positions;
};

}  // namespace racon

#endif  // RACON_WINDOW_HPP_
