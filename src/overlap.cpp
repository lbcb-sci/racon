// Copyright (c) 2020 Robert Vaser

#include "overlap.hpp"

#include "edlib.h"  // NOLINT

namespace racon {

Overlap::Overlap(const biosoup::Overlap& other)
    : biosoup::Overlap(other),
      lhs_intervals(),
      rhs_intervals() {
}

Overlap& Overlap::operator=(const biosoup::Overlap& other) {
  lhs_id = other.lhs_id;
  lhs_begin = other.lhs_begin;
  lhs_end = other.lhs_end;
  rhs_id = other.rhs_id;
  rhs_begin = other.rhs_begin;
  rhs_end = other.rhs_end;
  score = other.score;
  strand = other.strand;
  alignment = other.alignment;
  return *this;
}

void Overlap::Align(
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& sequences,
    const std::vector<std::unique_ptr<biosoup::NucleicAcid>>& targets) {
  if (!alignment.empty()) {
    return;
  }
  if ((lhs_id < sequences.size() && sequences[lhs_id]->id != lhs_id) ||
      (lhs_id >= sequences.size())) {
    throw std::invalid_argument("[racon::Overlap::Align] error: "
        "missing query sequence");
  }
  if ((rhs_id < targets.size() && targets[rhs_id]->id != rhs_id) ||
      (rhs_id >= targets.size())) {
    throw std::invalid_argument("[racon::Overlap::Align] error: "
        "missing target sequence");
  }

  std::string query = sequences[lhs_id]->InflateData(lhs_begin, lhs_end - lhs_begin);  // NOLINT
  std::string target = targets[rhs_id]->InflateData(rhs_begin, rhs_end - rhs_begin);  // NOLINT

  EdlibAlignResult result = edlibAlign(
      query.c_str(), query.size(),
      target.c_str(), target.size(),
      edlibNewAlignConfig(
          -1,
          EDLIB_MODE_NW,
          EDLIB_TASK_PATH,
          nullptr,
          0));
  if (result.status == EDLIB_STATUS_OK) {
    char* ret = edlibAlignmentToCigar(
        result.alignment,
        result.alignmentLength,
        EDLIB_CIGAR_STANDARD);
    alignment = ret;
    free(ret);
  } else {
    throw std::invalid_argument(
        "[racon::Overlap::Align] error: unable to align sequences "
        + std::to_string(lhs_id) + " and "
        + std::to_string(rhs_id));
  }
  edlibFreeAlignResult(result);
}

void Overlap::FindIntervals(std::uint32_t w) {
  if (!lhs_intervals.empty()) {
    return;
  }
  if (alignment.empty()) {
    throw std::logic_error(
        "[racon::Overlap::FindIntervals] error: missing alignment");
  }

  std::vector<std::int32_t> window_ends;
  for (std::uint32_t i = 0; i < rhs_end; i += w) {
    if (i > rhs_begin) {
      window_ends.emplace_back(i - 1);
    }
  }
  window_ends.emplace_back(rhs_end - 1);

  std::uint32_t k = 0;
  bool found_first = false;
  std::pair<std::uint32_t, std::uint32_t> first = {0, 0};
  std::pair<std::uint32_t, std::uint32_t> last = {0, 0};

  std::int32_t q_ptr = lhs_begin - 1;
  std::int32_t t_ptr = rhs_begin - 1;

  for (std::uint32_t i = 0, j = 0; i < alignment.size(); ++i) {
    switch (alignment[i]) {
      case 'M': case '=': case 'X': {
        std::uint32_t l = 0, num_bases = atoi(&alignment[j]);
        j = i + 1;
        while (l < num_bases) {
          ++q_ptr;
          ++t_ptr;

          if (!found_first) {
            found_first = true;
            first = {q_ptr, t_ptr};
          }
          last = {q_ptr, t_ptr};
          if (t_ptr == window_ends[k]) {
            if (found_first) {
              lhs_intervals.emplace_back(first.first, last.first + 1);
              rhs_intervals.emplace_back(first.second, last.second + 1);
            }
            found_first = false;
            ++k;
          }
          ++l;
        }
        break;
      }
      case 'I': {
        q_ptr += atoi(&alignment[j]);
        j = i + 1;
        break;
      }
      case 'D': case 'N': {
        std::uint32_t l = 0, num_bases = atoi(&alignment[j]);
        j = i + 1;
        while (l < num_bases) {
          ++t_ptr;
          if (t_ptr == window_ends[k]) {
            if (found_first) {
              lhs_intervals.emplace_back(first.first, last.first + 1);
              rhs_intervals.emplace_back(first.second, last.second + 1);
            }
            found_first = false;
            ++k;
          }
          ++l;
        }
        break;
      }
      case 'S': case 'H': case 'P': {
        j = i + 1;
        break;
      }
      default: {
        break;
      }
    }
  }

  std::string{}.swap(alignment);
}

}  // namespace racon
