/*!
 * @file window.cpp
 *
 * @brief Window class source file
 */

#include "window.hpp"

#include <algorithm>
#include <stdexcept>

#include "spoa/spoa.hpp"

namespace racon {

Window::Window(
    std::uint64_t id,
    std::uint32_t rank,
    WindowType type,
    const char* backbone, std::uint32_t backbone_len,
    const char* quality, uint32_t quality_len)
    : id_(id),
      rank_(rank),
      type_(type),
      consensus_(),
      sequences_(1, std::make_pair(backbone, backbone_len)),
      qualities_(1, std::make_pair(quality, quality_len)),
      positions_(1, std::make_pair(0, 0)) {
}

void Window::AddLayer(
    const char* sequence, std::uint32_t sequence_len,
    const char* quality, std::uint32_t quality_len,
    std::uint32_t begin,
    std::uint32_t end) {

  if (quality != nullptr && sequence_len != quality_len) {
    throw std::invalid_argument(
        "[racon::Window::AddLayer] error: unequal quality size");
  }
  if (begin >= end ||
      begin > sequences_.front().second ||
      end > sequences_.front().second) {
    throw std::invalid_argument(
        "[racon::Window::AddLayer] error: invalid positions");
  }

  sequences_.emplace_back(sequence, sequence_len);
  qualities_.emplace_back(quality, quality_len);
  positions_.emplace_back(begin, end);
}

bool Window::GenerateConsensus(
    std::shared_ptr<spoa::AlignmentEngine> alignment_engine,
    bool trim) {

  if (sequences_.size() < 3) {
    consensus_ = std::string(
        sequences_.front().first, sequences_.front().second);
    return false;
  }

  auto graph = spoa::createGraph();
  graph->add_alignment(
      spoa::Alignment(),
      sequences_.front().first, sequences_.front().second,
      qualities_.front().first, qualities_.front().second);

  std::vector<std::uint32_t> rank;
  rank.reserve(sequences_.size());
  for (std::uint32_t i = 0; i < sequences_.size(); ++i) {
    rank.emplace_back(i);
  }

  std::stable_sort(rank.begin() + 1, rank.end(),
      [&] (std::uint32_t lhs, std::uint32_t rhs) {
        return positions_[lhs].first < positions_[rhs].first;
      });

  std::uint32_t offset = 0.01 * sequences_.front().second;
  for (std::uint32_t j = 1; j < sequences_.size(); ++j) {
    std::uint32_t i = rank[j];

    spoa::Alignment alignment;
    if (positions_[i].first < offset &&
        positions_[i].second > sequences_.front().second - offset) {
      alignment = alignment_engine->align(
          sequences_[i].first, sequences_[i].second,
          graph);
    } else {
      std::vector<int32_t> mapping;
      auto subgraph = graph->subgraph(
          positions_[i].first, positions_[i].second,
          mapping);
      alignment = alignment_engine->align(
          sequences_[i].first, sequences_[i].second,
          subgraph);
      subgraph->update_alignment(alignment, mapping);
    }

    if (qualities_[i].first == nullptr) {
      graph->add_alignment(
          alignment,
          sequences_[i].first, sequences_[i].second);
    } else {
      graph->add_alignment(
        alignment,
        sequences_[i].first, sequences_[i].second,
        qualities_[i].first, qualities_[i].second);
      }
  }

  std::vector<std::uint32_t> coverages;
  consensus_ = graph->generate_consensus(coverages);

  if (type_ == WindowType::kTGS && trim) {
    std::uint32_t average_coverage = (sequences_.size() - 1) / 2;

    std::int32_t begin = 0, end = consensus_.size() - 1;
    for (; begin < static_cast<std::int32_t>(consensus_.size()); ++begin) {
      if (coverages[begin] >= average_coverage) {
        break;
      }
    }
    for (; end >= 0; --end) {
      if (coverages[end] >= average_coverage) {
        break;
      }
    }

    if (begin < end) {
      consensus_ = consensus_.substr(begin, end - begin + 1);
    }
  }

  return true;
}

}  // namespace racon
