// Copyright (c) 2020 Robert Vaser

#ifndef RACON_OVERLAP_HPP_
#define RACON_OVERLAP_HPP_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include <string>
#include <utility>

#include "biosoup/nucleic_acid.hpp"
#include "biosoup/overlap.hpp"

namespace racon {

struct Overlap: public biosoup::Overlap {
 public:
  Overlap() = default;

  explicit Overlap(const biosoup::Overlap& other);
  Overlap& operator=(const biosoup::Overlap& other);

  Overlap(const Overlap&) = default;
  Overlap& operator=(const Overlap&) = default;

  Overlap(Overlap&&) = default;
  Overlap& operator=(Overlap&&) = default;

  ~Overlap() = default;

  std::uint32_t length() const {
    return std::max(lhs_end - lhs_begin, rhs_end - rhs_begin);
  }

  void Align(
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
      const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets);

  // find window points from cigar string
  void FindIntervals(std::uint32_t w);

  std::vector<std::pair<std::uint32_t, std::uint32_t>> lhs_intervals;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> rhs_intervals;
};

}  // namespace racon

#endif  // RACON_OVERLAP_HPP_
