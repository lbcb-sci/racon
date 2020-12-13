// Copyright (c) 2020 Robert Vaser

#ifndef RACON_WINDOW_HPP_
#define RACON_WINDOW_HPP_

#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "biosoup/nucleic_acid.hpp"
#include "spoa/alignment_engine.hpp"

namespace racon {

enum class WindowType {
  kNGS,  // Next Generation Sequencing
  kTGS  // Third Generation Sequencing
};

struct Window {
 public:
  Window(std::uint64_t id, std::uint32_t rank, WindowType type);

  Window(const Window&) = default;
  Window& operator=(const Window&) = default;

  Window(Window&&) = default;
  Window& operator=(Window&&) = default;

  ~Window() = default;

  void AddLayer(
      std::uint32_t sequence_id,
      std::uint32_t sequence_begin,
      std::uint32_t sequence_end,
      std::uint32_t window_begin,
      std::uint32_t window_end);

  void Fill(
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences);

  void GenerateConsensus(
      std::shared_ptr<spoa::AlignmentEngine> alignment_engine,
      bool trim);

  std::uint64_t id;
  std::uint32_t rank;
  WindowType type;
  bool status;
  std::string consensus;
  std::vector<std::uint32_t> sequences_ids;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> sequences_intervals;
  std::vector<std::string> sequences;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> positions;
};

}  // namespace racon

#endif  // RACON_WINDOW_HPP_
