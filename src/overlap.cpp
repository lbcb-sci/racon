/*!
 * @file overlap.cpp
 *
 * @brief Overlap class source file
 */

#include <algorithm>

#include "edlib.h"  // NOLINT

#include "overlap.hpp"

namespace racon {

Overlap::Overlap(
    std::uint32_t q_id, std::uint32_t q_begin, std::uint32_t q_end,
    std::uint32_t t_id, std::uint32_t t_begin, std::uint32_t t_end,
    std::uint32_t strand)
    : q_id(q_id),
      q_begin(q_begin),
      q_end(q_end),
      t_id(t_id),
      t_begin(t_begin),
      t_end(t_end),
      strand(strand),
      cigar(),
      break_points() {
}

std::uint32_t Overlap::Length() const {
  return std::max(t_end - t_begin, q_end - q_begin);
}

void Overlap::FindBreakPoints(
    const std::vector<std::unique_ptr<biosoup::Sequence>>& targets,
    const std::vector<std::unique_ptr<biosoup::Sequence>>& sequences,
    std::uint32_t w) {

  if (!break_points.empty()) {
    return;
  }
  if (cigar.empty()) {
    const char* q = &(sequences[q_id]->data[q_begin]);
    const char* t = &(targets[t_id]->data[t_begin]);
    Align(q, q_end - q_begin, t, t_end - t_begin);
  }

  FindBreakPoints(w);
}

void Overlap::FindBreakPoints(std::uint32_t w) {
  std::vector<std::int32_t> window_ends;
  for (std::uint32_t i = 0; i < t_end; i += w) {
    if (i > t_begin) {
      window_ends.emplace_back(i - 1);
    }
  }
  window_ends.emplace_back(t_end - 1);

  std::uint32_t k = 0;
  bool found_first_match = false;
  std::pair<std::uint32_t, std::uint32_t> first_match = {0, 0};
  std::pair<std::uint32_t, std::uint32_t> last_match = {0, 0};

  std::int32_t q_ptr = q_begin - 1;
  std::int32_t t_ptr = t_begin - 1;

  for (std::uint32_t i = 0, j = 0; i < cigar.size(); ++i) {
    if (cigar[i] == 'M' || cigar[i] == '=' || cigar[i] == 'X') {
      std::uint32_t l = 0, num_bases = atoi(&cigar[j]);
      j = i + 1;
      while (l < num_bases) {
        ++q_ptr;
        ++t_ptr;

        if (!found_first_match) {
          found_first_match = true;
          first_match.first = t_ptr;
          first_match.second = q_ptr;
        }
        last_match.first = t_ptr + 1;
        last_match.second = q_ptr + 1;
        if (t_ptr == window_ends[k]) {
          if (found_first_match) {
              break_points.emplace_back(first_match);
              break_points.emplace_back(last_match);
          }
          found_first_match = false;
          ++k;
        }
        ++l;
      }
    } else if (cigar[i] == 'I') {
      q_ptr += atoi(&cigar[j]);
      j = i + 1;
    } else if (cigar[i] == 'D' || cigar[i] == 'N') {
      std::uint32_t l = 0, num_bases = atoi(&cigar[j]);
      j = i + 1;
      while (l < num_bases) {
        ++t_ptr;
        if (t_ptr == window_ends[k]) {
          if (found_first_match) {
            break_points.emplace_back(first_match);
            break_points.emplace_back(last_match);
          }
          found_first_match = false;
          ++k;
        }
        ++l;
      }
    } else if (cigar[i] == 'S' || cigar[i] == 'H' || cigar[i] == 'P') {
      j = i + 1;
    }
  }

  std::string().swap(cigar);
}

void Overlap::Align(
    const char* q, std::uint32_t q_length,
    const char* t, std::uint32_t t_length) {

  EdlibAlignResult result = edlibAlign(
      q, q_length,
      t, t_length,
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
    cigar = ret;
    free(ret);
  } else {
    throw std::invalid_argument(
      "[racon::Overlap::FindBreakingPoints] error: edlib unable to align "
      + std::to_string(q_id) + " and " + std::to_string(t_id));
  }

  edlibFreeAlignResult(result);
}

}  // namespace racon
